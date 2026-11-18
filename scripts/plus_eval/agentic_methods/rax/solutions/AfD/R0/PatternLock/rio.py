"""Defensive parsing of RoboMME MCP replies (no robot motion here)."""
import base64
import io
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

WS = Path('/workspace')


def blocks(result):
    if isinstance(result, dict):
        return result.get('content') or []
    return []


def meta_of(result):
    """Merge all JSON text blocks (and structuredContent) into one dict."""
    meta = {}
    if isinstance(result, dict) and isinstance(result.get('structuredContent'), dict):
        meta.update(result['structuredContent'])
    for b in blocks(result):
        if b.get('type') == 'text':
            try:
                j = json.loads(b.get('text', ''))
            except Exception:
                continue
            if isinstance(j, dict):
                for k, v in j.items():
                    meta.setdefault(k, v)
            elif isinstance(j, list):
                meta.setdefault('_list', j)
    return meta


def texts_of(result):
    return [b.get('text', '') for b in blocks(result) if b.get('type') == 'text']


def find_all(obj, key, depth=0, out=None):
    """Breadth-ordered list of values stored under `key` anywhere in obj."""
    if out is None:
        out = []
    level = [obj]
    while level and depth < 12:
        nxt = []
        for o in level:
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == key:
                        out.append(v)
                    if isinstance(v, (dict, list)):
                        nxt.append(v)
            elif isinstance(o, list):
                for v in o:
                    if isinstance(v, (dict, list)):
                        nxt.append(v)
        level = nxt
        depth += 1
    return out


def find_first(obj, key, default=None, pred=None):
    for v in find_all(obj, key):
        if pred is None or pred(v):
            return v
    return default


def is_vec(v, n=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def state_of(meta):
    st = find_first(meta, 'state', pred=lambda v: is_vec(v, 8))
    if st is None:
        st = find_first(meta, 'state', pred=lambda v: is_vec(v) and len(v) >= 6)
    return None if st is None else np.asarray(st, float)


def step_id_of(meta):
    v = find_first(meta, 'step_id', pred=lambda v: isinstance(v, int))
    return v


def flag(meta, key):
    vals = find_all(meta, key)
    return any(v is True for v in vals)


def decode_images(result):
    """Return list of (label_hint, PIL.Image) for image blocks, in order."""
    out = []
    hint = ''
    for b in blocks(result):
        t = b.get('type')
        if t == 'text':
            hint = b.get('text', '')[-200:]
        elif t == 'image':
            try:
                data = base64.b64decode(b.get('data', ''))
                im = Image.open(io.BytesIO(data))
                im.load()
                lab = ''
                for key in ('camera', 'name', 'title', 'label'):
                    if isinstance(b.get(key), str):
                        lab = b[key]
                ann = b.get('annotations') or b.get('_meta') or {}
                if isinstance(ann, dict):
                    for key in ('camera', 'name', 'title', 'label'):
                        if isinstance(ann.get(key), str):
                            lab = ann[key]
                out.append((lab or hint, im))
            except Exception:
                continue
    return out


def rgb_pair(result, ref_front=None):
    """Identify front and wrist RGB arrays among inline images."""
    ims = decode_images(result)
    rgbs = []
    for lab, im in ims:
        if im.mode in ('RGB', 'RGBA', 'P', 'L') and im.mode != 'I;16' and im.mode != 'I':
            a = np.asarray(im.convert('RGB'))
            # skip depth-like single-channel images
            if im.mode == 'L':
                continue
            rgbs.append((lab.lower() if isinstance(lab, str) else '', a))
    front = wrist = None
    for lab, a in rgbs:
        tail = lab[-60:]
        if 'wrist' in tail and wrist is None:
            wrist = a
        elif 'front' in tail and front is None:
            front = a
    if front is None or wrist is None:
        cand = [a for _, a in rgbs]
        if len(cand) >= 2:
            if ref_front is not None and cand[0].shape == ref_front.shape and cand[1].shape == ref_front.shape:
                d0 = np.abs(cand[0].astype(int) - ref_front.astype(int)).mean()
                d1 = np.abs(cand[1].astype(int) - ref_front.astype(int)).mean()
                front, wrist = (cand[0], cand[1]) if d0 <= d1 else (cand[1], cand[0])
            else:
                front, wrist = cand[0], cand[1]
        elif len(cand) == 1:
            front = cand[0]
    return front, wrist


def manifest_path_of(meta):
    seq = find_first(meta, 'frame_sequence', pred=lambda v: isinstance(v, dict))
    p = None
    if isinstance(seq, dict):
        p = seq.get('manifest_path')
    if p is None:
        p = find_first(meta, 'manifest_path', pred=lambda v: isinstance(v, str))
    return p


def load_manifest(path):
    p = Path(path)
    if not p.is_absolute():
        p = WS / p
    return json.loads(p.read_text()), p


def frame_files(frame):
    out = {}
    for f in frame.get('files', []) or []:
        cam = f.get('camera')
        kind = f.get('kind')
        if cam and kind and f.get('path'):
            out[(cam, kind)] = f
    return out


def resolve(path, manifest_file=None):
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    q = WS / p
    if q.exists():
        return q
    if manifest_file is not None:
        r = Path(manifest_file).parent / p
        if r.exists():
            return r
    return q


def load_rgb(entry, manifest_file=None):
    with Image.open(resolve(entry['path'], manifest_file)) as im:
        return np.asarray(im.convert('RGB'))


def load_depth(entry, manifest_file=None):
    with Image.open(resolve(entry['path'], manifest_file)) as im:
        a = np.asarray(im)
    a = a.astype(np.float64)
    if a.ndim == 3:
        a = a[..., 0]
    mpu = entry.get('meters_per_unit', 0.001) or 0.001
    return a * float(mpu)


# ------------------------------------------------------------ rotations
def rotvec_to_mat(r):
    r = np.asarray(r, float)
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def mat_to_rotvec(R):
    R = np.asarray(R, float)
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(k))
        if k[i] > 1e-9:
            k = A[:, i] / k[i]
        k = k / (np.linalg.norm(k) + 1e-12)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ------------------------------------------------------------ calibration
def _intr(v):
    """Return (fx, fy, cx, cy) from many possible encodings."""
    if isinstance(v, dict):
        for k in ('fx',):
            if k in v and 'fy' in v:
                return float(v['fx']), float(v['fy']), float(v.get('cx', v.get('ppx', 128))), float(v.get('cy', v.get('ppy', 128)))
        for k in ('K', 'matrix', 'intrinsic_matrix', 'intrinsics', 'camera_matrix', 'k'):
            if k in v:
                return _intr(v[k])
        return None
    if isinstance(v, (list, tuple)):
        a = np.asarray(v, float)
        if a.shape == (3, 3):
            return a[0, 0], a[1, 1], a[0, 2], a[1, 2]
        if a.size == 9:
            a = a.reshape(3, 3)
            return a[0, 0], a[1, 1], a[0, 2], a[1, 2]
        if a.size == 4:
            return tuple(a)
    return None


def _pose(v):
    """Return (R_cam_to_world, t) from many encodings (OpenGL camera axes)."""
    if isinstance(v, dict):
        pos = None
        for k in ('position', 'p', 'translation', 't', 'position_m'):
            if k in v and is_vec(v[k], 3):
                pos = np.asarray(v[k], float)
        q = None
        for k in ('quaternion_wxyz', 'wxyz', 'quaternion', 'q', 'orientation'):
            if k in v and is_vec(v[k], 4):
                q = v[k]
        if pos is not None and q is not None:
            return quat_wxyz_to_mat(q), pos
        for k in ('matrix', 'pose', 'T', 'cam2world', 'model_matrix'):
            if k in v:
                return _pose(v[k])
        return None
    if isinstance(v, (list, tuple)):
        a = np.asarray(v, float)
        if a.size == 16:
            a = a.reshape(4, 4)
            return a[:3, :3], a[:3, 3]
        if a.size == 7:
            return quat_wxyz_to_mat(a[3:]), a[:3]
    return None


def calibration(meta):
    """{camera: (fx,fy,cx,cy,R,t)} where available."""
    out = {}
    intr = find_first(meta, 'camera_intrinsics')
    poses = find_first(meta, 'camera_poses')
    for cam in ('front', 'wrist'):
        try:
            i = _intr(intr.get(cam)) if isinstance(intr, dict) else None
            p = _pose(poses.get(cam)) if isinstance(poses, dict) else None
            if i is not None and p is not None:
                out[cam] = (i, p)
        except Exception:
            pass
    return out


def backproject(cal, u, v, d, variant=0):
    """Pixel (u,v) with linear depth d -> world.  variant selects convention."""
    (fx, fy, cx, cy), (R, t) = cal
    off = 0.5 if variant in (1, 3) else 0.0
    if variant in (0, 1):  # OpenGL camera (-Z forward, +Y up)
        pc = np.array([(u + off - cx) / fx * d, -(v + off - cy) / fy * d, -d])
    else:  # OpenCV camera (+Z forward, +Y down)
        pc = np.array([(u + off - cx) / fx * d, (v + off - cy) / fy * d, d])
    return R @ pc + t


def project(cal, X, variant=0):
    (fx, fy, cx, cy), (R, t) = cal
    pc = R.T @ (np.asarray(X, float) - t)
    off = 0.5 if variant in (1, 3) else 0.0
    if variant in (0, 1):
        d = -pc[2]
        if d <= 1e-6:
            return None
        return np.array([fx * pc[0] / d + cx - off, -fy * pc[1] / d + cy - off])
    d = pc[2]
    if d <= 1e-6:
        return None
    return np.array([fx * pc[0] / d + cx - off, fy * pc[1] / d + cy - off])


def _xyz_from(it):
    if is_vec(it, 3):
        return np.asarray(it, float)
    if isinstance(it, dict):
        if all(k in it for k in ('x', 'y', 'z')) and is_vec([it['x'], it['y'], it['z']], 3):
            return np.asarray([it['x'], it['y'], it['z']], float)
        for k in ('position', 'world_position', 'world_position_m', 'world', 'point', 'xyz', 'position_m',
                  'world_point', 'world_xyz', 'point_world', 'world_m', 'p'):
            if k in it:
                v = _xyz_from(it[k])
                if v is not None:
                    return v
        for k, v in it.items():
            if k in ('rgb', 'color', 'pixel', 'uv', 'camera_position', 'camera'):
                continue
            if is_vec(v, 3):
                return np.asarray(v, float)
    return None


def p2p_points(result, n_expected=None):
    """Parse robomme_pixel_to_position reply -> list of (xyz or None)."""
    meta = meta_of(result)
    cands = []

    def walk(o, depth=0):
        if depth > 8:
            return
        if isinstance(o, dict):
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, list):
            if o and all(isinstance(x, (dict, list)) for x in o):
                cands.append(o)
            for v in o:
                if isinstance(v, (dict, list)):
                    walk(v, depth + 1)
    walk(meta)
    best, best_n = None, -1
    for lst in cands:
        vals = []
        for it in lst:
            ok = not (isinstance(it, dict) and it.get('valid') is False)
            v = _xyz_from(it)
            vals.append(v if ok else None)
        n = sum(v is not None for v in vals)
        bonus = 0.5 if (n_expected is not None and len(lst) == n_expected) else 0
        if n + bonus > best_n:
            best, best_n = vals, n + bonus
    return best or []


def backproject_all(cal, depth, variant=0):
    """World XYZ for every pixel (H, W, 3); invalid depth -> nan."""
    (fx, fy, cx, cy), (R, t) = cal
    d = np.asarray(depth, float)
    H, W = d.shape
    vv, uu = np.mgrid[0:H, 0:W].astype(float)
    off = 0.5 if variant in (1, 3) else 0.0
    if variant in (0, 1):
        pc = np.stack([(uu + off - cx) / fx * d, -(vv + off - cy) / fy * d, -d], -1)
    else:
        pc = np.stack([(uu + off - cx) / fx * d, (vv + off - cy) / fy * d, d], -1)
    X = pc @ R.T + t
    X[d <= 0] = np.nan
    return X
