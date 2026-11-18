"""Defensive parsing of MCP replies and geometry helpers (no task logic)."""
import base64
import io
import json
import math
import os
import re
import struct
import zlib

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

WORKSPACE = '/workspace'


# ---------------------------------------------------------------- JSON search
def iter_dicts(obj, depth=0):
    if depth > 12:
        return
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from iter_dicts(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from iter_dicts(v, depth + 1)


def find_key(obj, key, pred=None):
    """Breadth-first search for the first value under `key` satisfying pred."""
    queue = [obj]
    seen = 0
    while queue and seen < 20000:
        cur = queue.pop(0)
        seen += 1
        if isinstance(cur, dict):
            if key in cur and (pred is None or pred(cur[key])):
                return cur[key]
            queue.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            queue.extend(cur)
    return None


def is_vec(v, n):
    return isinstance(v, (list, tuple)) and len(v) == n and all(isinstance(x, (int, float)) for x in v)


# ---------------------------------------------------------------- reply
def text_blocks(result):
    out = []
    for b in (result or {}).get('content', []) or []:
        if isinstance(b, dict) and b.get('type') == 'text':
            out.append(b.get('text', ''))
    return out


def reply_meta(result):
    """Merge every JSON text block (first wins on conflicts)."""
    meta = {}
    extra = []
    for t in text_blocks(result):
        try:
            j = json.loads(t)
        except Exception:
            continue
        if isinstance(j, dict):
            if not meta:
                meta = j
            else:
                extra.append(j)
        else:
            extra.append(j)
    if extra:
        meta = dict(meta)
        meta['_extra'] = extra
    if isinstance(result, dict) and 'structuredContent' in result and isinstance(result['structuredContent'], dict):
        if not meta:
            meta = dict(result['structuredContent'])
        else:
            meta.setdefault('_structured', result['structuredContent'])
    return meta


def decode_png_bytes(data):
    """Decode PNG/JPEG bytes -> numpy array (HxWx3 uint8 or HxW uint16/uint8)."""
    if Image is not None:
        im = Image.open(io.BytesIO(data))
        im.load()
        if im.mode in ('I;16', 'I;16B', 'I;16L', 'I'):
            return np.array(im).astype(np.uint16 if im.mode != 'I' else np.int64)
        if im.mode in ('L', 'LA', 'P') and im.mode != 'P':
            return np.array(im.convert('L'))
        return np.array(im.convert('RGB'))
    return _png_fallback(data)


def _png_fallback(data):  # minimal PNG decoder (8-bit RGB/RGBA/gray, 16-bit gray)
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    pos, idat, w = 8, b'', None
    while pos < len(data):
        ln, typ = struct.unpack('>I4s', data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + ln]
        if typ == b'IHDR':
            w, h, bd, ct = struct.unpack('>IIBB', chunk[:10])
        elif typ == b'IDAT':
            idat += chunk
        pos += 12 + ln
    ch = {0: 1, 2: 3, 6: 4, 4: 2}[ct]
    bpp = ch * (bd // 8)
    raw = zlib.decompress(idat)
    stride = w * bpp
    out = np.zeros((h, stride), np.uint8)
    prev = np.zeros(stride, np.int32)
    p = 0
    for y in range(h):
        f = raw[p]
        line = np.frombuffer(raw[p + 1:p + 1 + stride], np.uint8).astype(np.int32)
        p += 1 + stride
        cur = np.zeros(stride, np.int32)
        for i in range(stride):
            a = cur[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            x = line[i]
            if f == 1:
                x += a
            elif f == 2:
                x += b
            elif f == 3:
                x += (a + b) // 2
            elif f == 4:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                x += a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            cur[i] = x & 255
        out[y] = cur
        prev = cur
    if bd == 16:
        arr = out.reshape(h, w, ch, 2)
        arr = (arr[..., 0].astype(np.uint16) << 8) | arr[..., 1]
    else:
        arr = out.reshape(h, w, ch)
    if ch == 1:
        return arr[..., 0]
    return arr[..., :3].astype(np.uint8)


def load_file(path):
    p = path if os.path.isabs(path) else os.path.join(WORKSPACE, path)
    with open(p, 'rb') as f:
        return decode_png_bytes(f.read())


def _label_from(d):
    s = json.dumps({k: v for k, v in d.items() if k not in ('data',)} if isinstance(d, dict) else d)[:400].lower()
    cam = 'front' if 'front' in s else ('wrist' if 'wrist' in s else None)
    kind = 'depth' if 'depth' in s else ('rgb' if ('rgb' in s or 'color' in s) else None)
    return cam, kind


def reply_images(result):
    """List of dicts {arr, camera, kind} from image content blocks (labels may be None)."""
    out = []
    last_text = ''
    for b in (result or {}).get('content', []) or []:
        if not isinstance(b, dict):
            continue
        if b.get('type') == 'text':
            last_text = b.get('text', '') or ''
            continue
        if b.get('type') != 'image' or not b.get('data'):
            continue
        try:
            arr = decode_png_bytes(base64.b64decode(b['data']))
        except Exception:
            continue
        cam, kind = _label_from({k: v for k, v in b.items() if k != 'data'})
        if cam is None and len(last_text) < 300:
            c2, k2 = _label_from(last_text)
            cam = cam or c2
            kind = kind or k2
        if kind is None:
            kind = 'rgb' if arr.ndim == 3 else 'depth'
        out.append(dict(arr=arr, camera=cam, kind=kind))
        last_text = ''
    return out


def meta_files(meta):
    """{(camera, kind): path} from any dict that has camera/kind/path keys."""
    files = {}
    for d in iter_dicts(meta):
        cam = d.get('camera')
        path = d.get('path') or d.get('file')
        if isinstance(cam, str) and isinstance(path, str):
            kind = d.get('kind') or ('depth' if 'depth' in path.lower() else 'rgb')
            files.setdefault((cam, kind), (path, d.get('meters_per_unit')))
    return files


# ---------------------------------------------------------------- geometry
def quat_wxyz_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec_to_mat(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def mat_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if abs(th - math.pi) < 1e-3:
        # near pi: axis from diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        i = int(np.argmax(ax))
        a = np.zeros(3)
        a[i] = ax[i]
        for j in range(3):
            if j != i:
                a[j] = R[i, j] / (2 * ax[i])
        return a / np.linalg.norm(a) * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap90(a):
    """Wrap angle to [-pi/4, pi/4) (square symmetry)."""
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


def backproject(u, v, depth_m, K, cam_pos, cam_quat_wxyz, convention='opengl'):
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    x = (u - cx) / fx * depth_m
    y = (v - cy) / fy * depth_m
    if convention == 'opengl':
        pc = np.array([x, -y, -depth_m])
    else:
        pc = np.array([x, y, depth_m])
    R = quat_wxyz_to_mat(cam_quat_wxyz)
    return R @ pc + np.asarray(cam_pos, float)


def parse_points(result, n):
    """Parse a robomme_pixel_to_position reply -> list of (xyz or None, spread)."""
    meta = reply_meta(result)
    cand_lists = []
    for d in iter_dicts(meta):
        for k, v in d.items():
            if isinstance(v, list) and len(v) == n and all(isinstance(e, dict) for e in v):
                cand_lists.append(v)
    if not cand_lists and isinstance(meta, list):
        cand_lists.append(meta)
    out = []
    if not cand_lists:
        return [(None, None)] * n
    lst = cand_lists[0]
    for e in lst:
        xyz = None
        for key in ('position', 'world_position', 'position_m', 'world', 'point', 'xyz', 'world_point', 'point_world'):
            v = e.get(key)
            if is_vec(v, 3):
                xyz = np.array(v, float)
                break
            if isinstance(v, dict):
                vv = [v.get(a) for a in ('x', 'y', 'z')]
                if all(isinstance(t, (int, float)) for t in vv):
                    xyz = np.array(vv, float)
                    break
        if xyz is None:
            for k, v in e.items():
                if is_vec(v, 3) and k not in ('pixel', 'uv', 'color', 'rgb'):
                    xyz = np.array(v, float)
                    break
        valid = e.get('valid', True)
        if valid is False:
            xyz = None
        spread = e.get('depth_spread_m')
        out.append((xyz, spread if isinstance(spread, (int, float)) else None))
    while len(out) < n:
        out.append((None, None))
    return out
