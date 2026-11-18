import json, math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation as R

WS = Path('/workspace')

def load_frame(fr):
    files = {(f['camera'], f['kind']): f['path'] for f in fr['files']}
    out = {}
    for cam in ('front', 'wrist'):
        if (cam, 'rgb') in files:
            out[cam + '_rgb'] = np.array(Image.open(WS / files[(cam, 'rgb')]).convert('RGB')).astype(np.float32)
        if (cam, 'depth') in files:
            out[cam + '_depth'] = np.array(Image.open(WS / files[(cam, 'depth')])).astype(np.float32) * 0.001
    out['K'] = fr.get('camera_intrinsics'); out['poses'] = fr.get('camera_poses'); out['state'] = fr.get('state')
    return out

def backproject(u, v, d, K, pose):
    M = np.array(K['matrix'], float); fx, fy, cx, cy = M[0, 0], M[1, 1], M[0, 2], M[1, 2]
    u = np.asarray(u, float); v = np.asarray(v, float); d = np.asarray(d, float)
    pc = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    w, x, y, z = pose['quaternion_wxyz']
    rot = R.from_quat([x, y, z, w])
    return rot.apply(pc.reshape(-1, 3)).reshape(pc.shape) + np.array(pose['position_m'])

def world_map(rgb_depth, K, pose):
    d = rgb_depth
    H, W = d.shape
    vv, uu = np.mgrid[0:H, 0:W]
    P = backproject(uu + 0.0, vv + 0.0, d, K, pose)
    P[d <= 0] = np.nan
    return P

def color_masks(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    eps = 1.0
    gr = g / (r + eps); br = b / (r + eps)
    mx = img.max(-1); mn = img.min(-1); sat = (mx - mn) / (mx + eps)
    m = {}
    m['red'] = (r > 90) & (g < 0.35 * r) & (b < 0.35 * r)
    m['target'] = (b > r * 0.95) & (r > 120) & (b > 150) & (g < b * 0.9)
    m['teal'] = (g > r * 1.25) & (b > r * 1.2) & (g > 70) & (abs(g - b) < 0.12 * g) & (b < 175)
    m['pink'] = (r > 150) & (gr > 0.6) & (gr < 0.8) & (br > 0.48) & (br < 0.75) & (b < g)
    m['yellow'] = (r > 120) & (gr > 0.75) & (br < 0.45) & (g > 100)
    m['beige'] = (r > 140) & (gr > 0.74) & (gr < 0.9) & (br > 0.48) & (br < 0.72)
    m['dark'] = (mx < 110) & (b >= r) & (sat < 0.35) & (mx > 40)
    m['ramp'] = (b > r * 1.4) & (b > 140) & (g > r * 1.2)
    m['white'] = (mn > 150) & (sat < 0.15)
    return m

def blobs(mask, min_px=4):
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) >= min_px:
            out.append({'u': xs.mean(), 'v': ys.mean(), 'n': len(xs), 'px': (xs, ys)})
    out.sort(key=lambda b: -b['n'])
    return out

def blob_world(b, P):
    xs, ys = b['px']
    pts = P[ys, xs]
    pts = pts[~np.isnan(pts).any(-1)]
    return pts

def table_mask(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    gr = g / (r + 1); br = b / (r + 1)
    return (r > 90) & (gr > 0.48) & (gr < 0.70) & (br > 0.2) & (br < 0.48)

def pink_class(pts):
    """classify a pink tip blob from its world points"""
    if len(pts) < 4: return 'ball', (0, 0)
    xy = pts[:, :2]; mu = xy.mean(0)
    ev = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov((xy - mu).T)), 0)) * 4
    minor, major = float(ev[0]), float(ev[1])
    if minor >= 0.045 and major / max(minor, 1e-6) < 1.45: return 'scoop', (minor, major)
    if major >= 0.06: return 'hook', (minor, major)
    return 'ball', (minor, major)

def moved_objects(imgs, cats=('pink', 'yellow', 'beige', 'dark')):
    """imgs: list of front RGB frames of the video (first = initial scene).
    returns per-category list of (blob, max fraction of pixels that became table)"""
    m0 = color_masks(imgs[0]); res = {}
    tabs = [table_mask(im) for im in imgs[1:]]
    for c in cats:
        lst = []
        for b in blobs(m0[c], 6)[:6]:
            xs, ys = b['px']
            fr = max([t[ys, xs].mean() for t in tabs] + [0.0])
            lst.append((b, float(fr)))
        res[c] = lst
    return res
