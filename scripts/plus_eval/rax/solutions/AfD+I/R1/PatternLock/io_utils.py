"""Tolerant parsing of RAX MCP replies (JSON text block + image blocks + published files)."""
import base64, io, json, os, time
import numpy as np
from PIL import Image

WS = '/workspace'
OUT = '/workspace/output'

class Logger:
    def __init__(self, path=os.path.join(OUT, 'controller_log.txt')):
        self.t0 = time.time()
        try:
            os.makedirs(OUT, exist_ok=True)
            self.f = open(path, 'a', buffering=1)
        except Exception:
            self.f = None

    def __call__(self, *args):
        s = '[%7.1fs] ' % (time.time() - self.t0) + ' '.join(str(a) for a in args)
        try:
            print(s, flush=True)
        except Exception:
            pass
        if self.f:
            try:
                self.f.write(s + '\n')
            except Exception:
                pass

    def dump(self, name, obj, limit=200000):
        try:
            with open(os.path.join(OUT, name), 'w') as f:
                txt = json.dumps(obj, default=str, indent=1)
                f.write(txt[:limit])
        except Exception as e:
            self('dump failed', name, e)

def meta_of(result):
    """First JSON text block (or structuredContent)."""
    if isinstance(result, dict):
        sc = result.get('structuredContent')
        for b in result.get('content', []) or []:
            if b.get('type') == 'text':
                try:
                    return json.loads(b.get('text', ''))
                except Exception:
                    continue
        if isinstance(sc, dict):
            return sc
    return {}

def walk(obj, path=()):
    """Yield (path, key, value) for all dict entries recursively."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from walk(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + (i,))

def find_key(obj, names):
    for _, k, v in walk(obj):
        if k in names:
            return v
    return None

def find_all(obj, name):
    return [v for _, k, v in walk(obj) if k == name]

def _as_state(st):
    if isinstance(st, dict):
        for k in ('values', 'vector', 'state', 'data'):
            if isinstance(st.get(k), list):
                st = st[k]; break
    if isinstance(st, list) and len(st) >= 6 and all(isinstance(x, (int, float)) for x in st[:6]):
        return np.array(st, float)
    return None

def get_state(meta):
    if isinstance(meta, dict) and 'state' in meta:
        v = _as_state(meta['state'])
        if v is not None:
            return v
    for st in find_all(meta, 'state'):
        v = _as_state(st)
        if v is not None:
            return v
    return None

def get_step_id(meta):
    v = meta.get('step_id') if isinstance(meta, dict) else None
    if v is None:
        v = find_key(meta, ('step_id',))
    return int(v) if isinstance(v, (int, float)) else None

def episode_flags(meta):
    """(ended, complete). Uses the documented 'episode_ended' key when present anywhere;
    generic terminated/truncated keys only as a fallback."""
    vals = find_all(meta, 'episode_ended')
    if vals:
        ended = any(v is True for v in vals)
    else:
        ended = any(v is True for k in ('terminated', 'truncated') for v in find_all(meta, k))
    complete = any(v is True for v in find_all(meta, 'task_complete'))
    return ended, complete

def remaining_steps(meta):
    lim = find_key(meta, ('episode_limits',))
    if isinstance(lim, dict):
        for k in ('remaining_steps', 'steps_remaining', 'remaining'):
            if isinstance(lim.get(k), (int, float)):
                return int(lim[k])
        mx = None; used = None
        for k in ('max_steps', 'step_limit', 'max_episode_steps', 'limit'):
            if isinstance(lim.get(k), (int, float)):
                mx = lim[k]
        for k in ('steps', 'step', 'elapsed_steps', 'steps_used', 'step_id'):
            if isinstance(lim.get(k), (int, float)):
                used = lim[k]
        if mx is not None and used is not None:
            return int(mx - used)
    return None

def _decode_image_bytes(data):
    im = Image.open(io.BytesIO(base64.b64decode(data)))
    im.load()
    return im

def _classify(im):
    """('rgb', array) or ('depth', array uint16/float)."""
    if im.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
        return 'depth', np.array(im)
    if im.mode in ('RGB', 'RGBA', 'P'):
        return 'rgb', np.array(im.convert('RGB'))
    if im.mode == 'L':
        return 'gray', np.array(im)
    return 'rgb', np.array(im.convert('RGB'))

MANIFEST_DIR = [None]

def load_path(p):
    q = p if os.path.isabs(p) else os.path.join(WS, p)
    if not os.path.exists(q) and MANIFEST_DIR[0] and not os.path.isabs(p):
        q = os.path.join(MANIFEST_DIR[0], p)
    im = Image.open(q); im.load()
    return im

def images_from_files(files):
    """files: list of {camera, kind, path}. Returns dict {(camera, kind): array}."""
    out = {}
    for f in files or []:
        try:
            cam = str(f.get('camera', '')).lower(); kind = str(f.get('kind', '')).lower()
            im = load_path(f['path'])
            cls, arr = _classify(im)
            if kind == 'depth' or cls == 'depth':
                out[(cam, 'depth')] = arr
            else:
                out[(cam, 'rgb')] = np.array(im.convert('RGB'))
        except Exception:
            continue
    return out

def images_from_result(result, meta, log=None):
    """Return dict {(camera, kind): array} for the reply's final frame, best effort.
    Inline image blocks (the final frame) take priority over published file listings.
    Extra keys: '_inline_rgb' (list of inline RGB arrays in order), '_inline_labeled' (bool)."""
    inline = {}
    files = {}
    try:
        frames = find_all(meta, 'frames')
        best = None
        for fl in frames:
            if isinstance(fl, list) and fl and isinstance(fl[-1], dict) and 'files' in fl[-1]:
                best = fl[-1]['files']
        if best is None:
            for _, k, v in walk(meta):
                if k == 'files' and isinstance(v, list) and v and isinstance(v[0], dict) and 'path' in v[0]:
                    best = v
        if best:
            files = images_from_files(best)
    except Exception as e:
        if log: log('file image parse failed', e)
    order = []
    try:
        blocks = result.get('content', []) if isinstance(result, dict) else []
        last_text = ''
        for b in blocks:
            t = b.get('type')
            if t == 'text':
                last_text = b.get('text', '')[:400].lower()
            elif t == 'image' and b.get('data'):
                lab = ' '.join([last_text, json.dumps(b.get('annotations', {})), json.dumps(b.get('_meta', {})),
                                str(b.get('name', '')), str(b.get('uri', ''))]).lower()
                im = _decode_image_bytes(b['data'])
                cls, arr = _classify(im)
                if cls == 'gray':
                    if 'depth' in lab:
                        cls = 'depth'
                    else:
                        cls, arr = 'rgb', np.repeat(arr[..., None], 3, 2)
                order.append((lab, cls, arr))
                last_text = ''
    except Exception as e:
        if log: log('inline image parse failed', e)
    rgb_unl = []; dep_unl = []; labeled = False
    for lab, cls, arr in order:
        f, w = 'front' in lab, 'wrist' in lab
        cam = 'front' if (f and not w) else ('wrist' if (w and not f) else None)
        kind = 'depth' if cls == 'depth' else 'rgb'
        if cam is None:
            (dep_unl if kind == 'depth' else rgb_unl).append(arr)
        else:
            labeled = True
            inline.setdefault((cam, kind), arr)
    for kind, lst in (('rgb', rgb_unl), ('depth', dep_unl)):
        for cam, arr in zip([c for c in ('front', 'wrist') if (c, kind) not in inline], lst):
            inline[(cam, kind)] = arr
    out = dict(files)
    out.update(inline)
    out['_inline_rgb'] = [arr for lab, cls, arr in order if cls != 'depth']
    out['_inline_labeled'] = labeled
    return out

def manifest_from_meta(meta):
    fs = find_key(meta, ('frame_sequence',))
    mp = None
    if isinstance(fs, dict):
        mp = fs.get('manifest_path') or find_key(fs, ('manifest_path',))
    if mp is None:
        mp = find_key(meta, ('manifest_path',))
    if mp:
        p = mp if os.path.isabs(mp) else os.path.join(WS, mp)
        try:
            with open(p) as f:
                return json.load(f), p
        except Exception:
            pass
    # fallback: newest manifest with video_demo frames under the run's published frames
    import glob
    cands = []
    for pat in ('/workspace/runtime/autonomous_controller/runs/*/frames/**/*.json',
                '/workspace/runtime/**/frames/**/manifest*.json'):
        cands += glob.glob(pat, recursive=True)
    best = None
    for p in sorted(set(cands), key=lambda q: os.path.getmtime(q), reverse=True)[:50]:
        try:
            with open(p) as f:
                m = json.load(f)
            fr = m.get('frames', []) if isinstance(m, dict) else []
            if any(isinstance(x, dict) and x.get('phase') == 'video_demo' for x in fr):
                best = (m, p); break
        except Exception:
            continue
    return best if best else (None, None)
