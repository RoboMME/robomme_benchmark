import json, os
import numpy as np
try:
    from scipy import ndimage
    _label = ndimage.label
except Exception:  # pragma: no cover - fallback without scipy
    ndimage = None
    def _label(mask):
        mask = np.asarray(mask, bool); lab = np.zeros(mask.shape, int); n = 0
        H, W = mask.shape
        for i0, j0 in zip(*np.nonzero(mask)):
            if lab[i0, j0]:
                continue
            n += 1; stack = [(i0, j0)]; lab[i0, j0] = n
            while stack:
                i, j = stack.pop()
                for a, b in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
                    if 0 <= a < H and 0 <= b < W and mask[a, b] and not lab[a, b]:
                        lab[a, b] = n; stack.append((a, b))
        return lab, n
from geom import backproject, load_rgb, load_depth, rgb_to_hsv

COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

def classify_hue(h):
    out = np.full(h.shape, -1, int)
    out[(h < 14) | (h >= 335)] = 0
    out[(h >= 14) & (h < 40)] = 1
    out[(h >= 40) & (h < 75)] = 2
    out[(h >= 75) & (h < 170)] = 3
    out[(h >= 190) & (h < 260)] = 4
    out[(h >= 260) & (h < 335)] = 5
    return out

def color_blobs(rgb, pts, min_px=4):
    """Return list of (color, n_px, xyz_median, (u,v)) of saturated blobs."""
    h, s, v = rgb_to_hsv(rgb)
    sat = (s > 0.72) & (v > 0.18)
    cls = classify_hue(h)
    res = []
    for ci, name in enumerate(COLORS):
        m = sat & (cls == ci)
        if m.sum() < min_px:
            continue
        lab, n = _label(m)
        for k in range(1, n + 1):
            mk = lab == k
            npx = int(mk.sum())
            if npx < min_px:
                continue
            p = pts[mk]
            p = p[np.isfinite(p).all(1)]
            if len(p) < max(2, npx // 3):
                continue
            xyz = np.median(p, 0)
            ys, xs = np.nonzero(mk)
            res.append((name, npx, xyz, (float(xs.mean()), float(ys.mean()))))
    return res

def frame_points(root, frame, cam):
    files = {(f['camera'], f['kind']): f['path'] for f in frame['files']}
    rgb = load_rgb(os.path.join(root, files[(cam, 'rgb')]))
    mpu = frame.get('depth', {}).get(cam, {}).get('meters_per_unit', 0.001)
    dep = load_depth(os.path.join(root, files[(cam, 'depth')]), mpu)
    K = frame['camera_intrinsics'][cam]['matrix']
    pose = frame['camera_poses'][cam]
    return rgb, backproject(dep, K, pose)

def track_cubes(manifest_path, root='/workspace', log=print, zmax=0.08):
    man = json.load(open(manifest_path))
    frames = sorted(man['frames'], key=lambda f: f['frame_index'])
    obs = []  # (frame_idx, color, xyz, cam, npx)
    for fr in frames:
        if fr['phase'] != 'video_demo':
            continue
        for cam in ('front', 'wrist'):
            try:
                rgb, pts = frame_points(root, fr, cam)
            except Exception as e:
                log(f'frame {fr["frame_index"]} {cam} load fail {e}')
                continue
            for name, npx, xyz, uv in color_blobs(rgb, pts):
                if not (-0.01 < xyz[2] < zmax):
                    continue
                obs.append((fr['frame_index'], name, xyz, cam, npx))
    # cluster per color in xy
    clusters = []  # dict color, pts list, frames
    for fi, name, xyz, cam, npx in obs:
        best, bd = None, 1e9
        for c in clusters:
            if c['color'] != name:
                continue
            d = np.linalg.norm(np.median(np.array(c['xyz'][-10:]), 0)[:2] - xyz[:2])
            if d < bd:
                best, bd = c, d
        if best is not None and bd < 0.03:
            best['xyz'].append(xyz); best['frames'].append(fi); best['cams'].add(cam)
        else:
            clusters.append({'color': name, 'xyz': [xyz], 'frames': [fi], 'cams': {cam}})
    out = []
    for c in clusters:
        nf = len(set(c['frames']))
        if nf < 3:
            continue
        P = np.array(c['xyz'])
        # last seen frames are most relevant (cube rests before being covered)
        last = max(c['frames'])
        sel = [p for p, f in zip(c['xyz'], c['frames']) if f >= last - 15]
        pos = np.median(np.array(sel), 0)
        out.append({'color': c['color'], 'pos': pos.tolist(), 'n_frames': nf,
                    'first': min(c['frames']), 'last': last, 'cams': sorted(c['cams'])})
    return out, frames

def white_points(rgb, pts, zmin=0.015, zmax=0.25):
    h, s, v = rgb_to_hsv(rgb)
    m = (s < 0.25) & (v > 0.55)
    p = pts[m]
    ok = np.isfinite(p).all(1) & (p[:, 2] > zmin) & (p[:, 2] < zmax)
    return p[ok]

def min_area_yaw(xy):
    """Return (yaw in [0,90) deg, center xy, extents) of min-area bounding rect."""
    best = None
    c0 = xy.mean(0)
    for a in np.arange(0, 90, 1.0):
        t = np.deg2rad(a)
        R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        q = (xy - c0) @ R.T
        lo = np.percentile(q, 3, 0); hi = np.percentile(q, 97, 0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            cq = (lo + hi) / 2
            center = c0 + cq @ R
            best = (area, a, center, hi - lo)
    return best[1], best[2], best[3]

def top_component(top, guess_xy, cell=0.003, maxd=0.03):
    """Split top-face points into xy-connected components; return the one nearest guess."""
    g = np.asarray(guess_xy)[:2]
    lo = top[:, :2].min(0) - cell
    ij = np.floor((top[:, :2] - lo) / cell).astype(int)
    shape = ij.max(0) + 2
    occ = np.zeros(shape, bool); occ[ij[:, 0], ij[:, 1]] = True
    if ndimage is not None:
        occ = ndimage.binary_closing(occ, iterations=1) | occ
    lab, n = _label(occ)
    if n <= 1:
        return top
    pl = lab[ij[:, 0], ij[:, 1]]
    best, bd = None, 1e9
    for k in range(1, n + 1):
        q = top[pl == k]
        if len(q) < 8:
            continue
        d = np.linalg.norm(q[:, :2].mean(0) - g)
        if d < bd:
            best, bd = q, d
    if best is None or bd > maxd:
        return top if best is None else best
    return best

def locate_container(points, guess_xy, radius=0.045):
    d = np.linalg.norm(points[:, :2] - np.asarray(guess_xy)[:2], axis=1)
    p = points[(d < radius) & (points[:, 2] < 0.095) & (points[:, 2] > 0.04)]
    if len(p) < 12:
        return None
    ztop = float(np.percentile(p[:, 2], 90))
    if not (0.055 < ztop < 0.09):
        ztop = 0.073
    top = p[np.abs(p[:, 2] - ztop) < 0.01]
    top = top_component(top, guess_xy)
    if len(top) < 8:
        return None
    yaw, center, ext = min_area_yaw(top[:, :2])
    return {'center': center.tolist(), 'ztop': float(ztop), 'yaw_deg': float(yaw),
            'ext': ext.tolist(), 'n': int(len(p)), 'n_top': int(len(top))}
