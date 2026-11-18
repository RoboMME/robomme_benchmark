"""Robot I/O: tolerant parsing of MCP replies, observation/step/pixel queries.

Every reply-shape assumption is defensive: the exact JSON layout was not observable
offline (R1, no exploration), so keys are searched recursively.
"""
import base64
import io
import json
import math
import os
import time

import numpy as np
from PIL import Image

WS = '/workspace'


def log(*a):
    msg = ' '.join(str(x) for x in a)
    line = '[%7.1f] %s' % (time.time() - LOG_T0[0], msg)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        os.makedirs(WS + '/output', exist_ok=True)
        with open(WS + '/output/controller_log.txt', 'a') as f:
            f.write(line + '\n')
    except Exception:
        pass


LOG_T0 = [time.time()]


# ---------------------------------------------------------------- JSON helpers
def iter_nodes(obj, depth=0):
    """Breadth-first walk over (key, value) pairs of nested dicts/lists."""
    queue = [(None, obj, 0)]
    while queue:
        k, v, d = queue.pop(0)
        yield k, v
        if d > 12:
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                queue.append((kk, vv, d + 1))
        elif isinstance(v, list) and len(v) < 5000:
            for vv in v:
                queue.append((None, vv, d + 1))


def find_key(obj, names, pred=None):
    if isinstance(names, str):
        names = [names]
    for k, v in iter_nodes(obj):
        if k in names and (pred is None or pred(v)):
            return v
    return None


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def is_vec(v, n):
    return isinstance(v, list) and len(v) == n and all(is_num(x) for x in v)


# ---------------------------------------------------------------- reply parsing
def _decode_image(data):
    raw = base64.b64decode(data)
    im = Image.open(io.BytesIO(raw))
    im.load()
    return im


def parse_reply(result):
    """Return dict(meta, rgb={'front':arr,'wrist':arr}, depth={...}, raw_images=[...])."""
    meta = {}
    content = []
    if isinstance(result, dict):
        content = result.get('content') or []
        sc = result.get('structuredContent')
        if isinstance(sc, dict):
            meta.update(sc)
    texts = []
    images = []  # (preceding_text, PIL image, block)
    last_text = ''
    for b in content:
        if not isinstance(b, dict):
            continue
        t = b.get('type')
        if t == 'text':
            txt = b.get('text', '')
            texts.append(txt)
            last_text = txt
            try:
                j = json.loads(txt)
                if isinstance(j, dict):
                    for kk, vv in j.items():
                        meta.setdefault(kk, vv)
            except Exception:
                pass
        elif t == 'image' and b.get('data'):
            try:
                im = _decode_image(b['data'])
                images.append((last_text, im, b))
            except Exception as e:
                log('image decode failed', e)
            last_text = ''
    rgb, depth = {}, {}
    unlabeled = []
    # explicit attachment list in metadata, aligned with image order
    att = None
    for k, v in iter_nodes(meta):
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v) and \
                all(('camera' in x) for x in v) and (len(v) == len(images) or
                                                      k in ('images', 'attachments', 'image_attachments')):
            att = v
            break
    for idx, (pre, im, blk) in enumerate(images):
        label = None
        kind = 'rgb' if im.mode in ('RGB', 'RGBA', 'P', 'L') else 'depth'
        for src in (blk.get('annotations'), blk.get('_meta'), blk.get('meta'), blk.get('name')):
            s = json.dumps(src).lower() if src is not None else ''
            if 'wrist' in s:
                label = 'wrist'
            elif 'front' in s:
                label = 'front'
            if 'depth' in s:
                kind = 'depth'
        if label is None and att is not None and idx < len(att):
            c = str(att[idx].get('camera', '')).lower()
            label = 'wrist' if 'wrist' in c else ('front' if 'front' in c else None)
            if 'depth' in str(att[idx].get('kind', '')).lower():
                kind = 'depth'
        if label is None and pre and len(pre) < 300:
            s = pre.lower()
            iw, ifr = s.rfind('wrist'), s.rfind('front')
            if iw >= 0 or ifr >= 0:
                label = 'wrist' if iw > ifr else 'front'
            if 'depth' in s:
                kind = 'depth'
        if kind == 'rgb' and label is None:
            a8 = np.asarray(im.convert('RGB'))
            if np.array_equal(a8[..., 0], a8[..., 1]) and np.array_equal(a8[..., 1], a8[..., 2]):
                kind = 'depth'  # grayscale attachment: treat as depth, never as a camera RGB
        if kind == 'rgb':
            arr = np.asarray(im.convert('RGB'))
            if label is None or label in rgb:
                unlabeled.append(arr)
            else:
                rgb[label] = arr
        else:
            arr = np.asarray(im).astype(np.float32)
            if label and label not in depth:
                depth[label] = arr
    guessed = False
    for arr in unlabeled:  # documented order guess: front first, then wrist
        for lab in ('front', 'wrist'):
            if lab not in rgb:
                rgb[lab] = arr
                guessed = True
                break
    return {'meta': meta, 'rgb': rgb, 'depth': depth, 'n_images': len(images), 'texts': texts,
            'guessed': guessed}


def _wood_bottom(arr):
    a = arr[-30:].astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return float(((r > 60) & (r - g > 20) & (g - b > 5) & (b < 150)).mean())


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(WS, p)


def rgb_from_paths(meta):
    """Fallback: rgb file paths listed directly in the metadata."""
    out = {}
    for k, v in iter_nodes(meta):
        if isinstance(v, dict) and isinstance(v.get('path'), str) and 'camera' in v:
            kind = str(v.get('kind', 'rgb')).lower()
            cam = str(v.get('camera', '')).lower()
            if 'depth' in kind or cam not in ('front', 'wrist') or cam in out:
                continue
            fp = _abs(v['path'])
            if os.path.exists(fp):
                try:
                    out[cam] = np.asarray(Image.open(fp).convert('RGB'))
                except Exception:
                    pass
    return out


def rgb_from_manifest(meta):
    """Fallback: last frame of a frame_sequence manifest."""
    mp = find_key(meta, 'manifest_path', lambda v: isinstance(v, str))
    if not mp:
        return {}
    p = mp if os.path.isabs(mp) else os.path.join(WS, mp)
    try:
        man = json.load(open(p))
        frames = man.get('frames') or []
        if not frames:
            return {}
        fr = max(frames, key=lambda f: f.get('frame_index', 0))
        out = {}
        for f in fr.get('files', []):
            if f.get('kind') == 'rgb':
                fp = f['path']
                fp = fp if os.path.isabs(fp) else os.path.join(WS, fp)
                if not os.path.exists(fp):
                    fp2 = os.path.join(os.path.dirname(p), f['path'])
                    fp = fp2 if os.path.exists(fp2) else fp
                out[f.get('camera')] = np.asarray(Image.open(fp).convert('RGB'))
        return out
    except Exception as e:
        log('manifest read failed', e)
        return {}


class Robot:
    """Thin stateful wrapper around ctx with tolerant parsing."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.step_id = None
        self.ended = False
        self.complete = False
        self.rgb = {}
        self.meta = {}
        self.instruction = None
        self.steps_used = 0
        self.remaining = None
        self.max_steps = None
        self.n_calls = 0
        self.swap = None

    # -------------------------------------------------- reply bookkeeping
    def _absorb(self, result, from_step=False):
        r = parse_reply(result)
        meta = r['meta']
        self.meta = meta
        st = find_key(meta, 'state', lambda v: is_vec(v, 8))
        if st is None:
            st = find_key(meta, ['proprio', 'robot_state', 'proprioception'], lambda v: is_vec(v, 8))
        if st is None and self.state is None:
            for k, v in iter_nodes(meta):
                if is_vec(v, 8):
                    st = v
                    log('WARNING: state taken from key %r' % k)
                    break
        if st is not None:
            self.state = np.array(st, dtype=float)
        sid = find_key(meta, 'step_id', is_num)
        if sid is None:
            sid = find_key(meta, ['observation_step_id', 'step', 'steps', 'elapsed_steps'], is_num)
        if sid is not None:
            self.step_id = int(sid)
        e = find_key(meta, 'episode_ended', lambda v: isinstance(v, bool))
        if e is not None:
            self.ended = self.ended or e
        tc = find_key(meta, 'task_complete', lambda v: isinstance(v, bool))
        if tc is not None:
            self.complete = self.complete or tc
        ins = find_key(meta, 'instruction', lambda v: isinstance(v, str) and len(v) > 10)
        if ins:
            self.instruction = ins
        lim = find_key(meta, 'episode_limits', lambda v: isinstance(v, dict))
        if lim:
            for k, v in lim.items():
                kl = k.lower()
                if is_num(v):
                    if 'remain' in kl:
                        self.remaining = int(v)
                    elif 'max' in kl or 'limit' in kl or 'total' in kl:
                        self.max_steps = int(v)
        rgb = r['rgb']
        if r.get('guessed') and 'front' in rgb and 'wrist' in rgb:
            if self.swap is None:
                wf, ww = _wood_bottom(rgb['front']), _wood_bottom(rgb['wrist'])
                self.swap = ww > wf + 0.15
                log('unlabeled camera images: bottom wood front-guess %.2f wrist-guess %.2f -> swap=%s' % (
                    wf, ww, self.swap))
            if self.swap:
                rgb = {'front': rgb['wrist'], 'wrist': rgb['front']}
        if ('front' not in rgb) or ('wrist' not in rgb):
            for k, v in rgb_from_paths(meta).items():
                rgb.setdefault(k, v)
        if ('front' not in rgb) or ('wrist' not in rgb):
            mrgb = rgb_from_manifest(meta)
            for k, v in mrgb.items():
                if k in ('front', 'wrist') and k not in rgb:
                    rgb[k] = v
        if rgb:
            self.rgb = rgb
        return r

    # -------------------------------------------------- tools
    def observe(self):
        self.n_calls += 1
        res = self.ctx.call('robomme_observe')
        self._absorb(res)
        return self

    def status(self):
        try:
            res = self.ctx.call('robomme_status')
            r = parse_reply(res)
            meta = r['meta']
            ins = find_key(meta, 'instruction', lambda v: isinstance(v, str) and len(v) > 10)
            if ins and not self.instruction:
                self.instruction = ins
            lim = find_key(meta, 'episode_limits', lambda v: isinstance(v, dict))
            if lim:
                for k, v in lim.items():
                    kl = k.lower()
                    if is_num(v):
                        if 'remain' in kl:
                            self.remaining = int(v)
                        elif 'max' in kl or 'limit' in kl or 'total' in kl:
                            self.max_steps = int(v)
            return meta
        except Exception as e:
            log('status failed', e)
            return {}

    def step(self, rows):
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        if not rows or self.ended:
            return self
        if self.remaining is not None:
            rows = rows[:max(1, self.remaining)]
        self.n_calls += 1
        prev = self.step_id
        res = self.ctx.call('robomme_step', actions=rows)
        self._absorb(res, from_step=True)
        if self.step_id is not None and prev is not None and self.step_id > prev:
            self.steps_used += self.step_id - prev
        else:
            self.steps_used += len(rows)
        if self.remaining is not None and find_key(self.meta, 'episode_limits') is None:
            self.remaining -= len(rows)
        return self

    def pixels_to_world(self, cam, uvs, radius=1):
        """uvs: list of (u, v) (u = column, v = row). Returns list of np.array(3) or None."""
        out = [None] * len(uvs)
        if self.step_id is None or not uvs:
            return out
        for s in range(0, len(uvs), 32):
            chunk = uvs[s:s + 32]
            sel = [{'camera': cam, 'u': float(u), 'v': float(v)} for (u, v) in chunk]
            res = None
            for attempt in range(2):
                try:
                    self.n_calls += 1
                    res = self.ctx.call('robomme_pixel_to_position', observation_step_id=int(self.step_id),
                                        selections=sel, depth_window_radius=int(radius))
                    break
                except Exception as e:
                    log('pixel_to_position failed (step_id %s): %s' % (self.step_id, str(e)[:300]))
                    if attempt == 0:
                        try:  # refresh the observation id (no physics) and retry once
                            self.observe()
                        except Exception:
                            pass
            if res is None:
                continue
            pts = parse_points(parse_reply(res)['meta'], len(chunk))
            for i, p in enumerate(pts):
                out[s + i] = p
        return out


_PREF = ['position', 'world_position', 'world_point', 'point', 'xyz', 'position_m', 'world', 'world_xyz', 'p']


def _point_of(el):
    if is_vec(el, 3):
        return np.array(el, dtype=float)
    if isinstance(el, dict):
        if el.get('valid') is False:
            return None
        if all(is_num(el.get(a)) for a in ('x', 'y', 'z')):
            return np.array([el['x'], el['y'], el['z']], dtype=float)
        for k in _PREF:
            v = el.get(k)
            if is_vec(v, 3):
                return np.array(v, dtype=float)
            if isinstance(v, dict) and all(is_num(v.get(a)) for a in 'xyz'):
                return np.array([v['x'], v['y'], v['z']], dtype=float)
        for k, v in el.items():
            kl = str(k).lower()
            if is_vec(v, 3) and 'camera' not in kl and 'pixel' not in kl and 'normal' not in kl and 'ray' not in kl:
                return np.array(v, dtype=float)
        for k, v in el.items():
            if isinstance(v, dict):
                p = _point_of(v)
                if p is not None:
                    return p
    return None


def parse_points(meta, n):
    best = None
    for k, v in iter_nodes(meta):
        if isinstance(v, list) and len(v) == n and all(isinstance(x, (dict, list)) for x in v):
            pts = [_point_of(x) for x in v]
            if any(p is not None for p in pts):
                best = pts
                break
    if best is None:
        return [None] * n
    return best
