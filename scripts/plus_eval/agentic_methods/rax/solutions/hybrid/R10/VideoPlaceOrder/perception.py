"""Video analysis for VideoPlaceOrder: placement events from robot state, target slots and swaps."""
import json, os, re
import numpy as np
from PIL import Image
from scipy import ndimage
from geom import Cam

COLORS = ('red', 'green', 'blue')

def load_frames(manifest_path, root='/workspace'):
    m = json.load(open(manifest_path))
    out = []
    for f in m['frames']:
        files = {(x['camera'], x['kind']): os.path.join(root, x['path']) for x in f.get('files', [])}
        out.append(dict(meta=f, files=files))
    return out

def read_rgb(path):
    return np.array(Image.open(path).convert('RGB'))

def read_depth(path):
    return np.array(Image.open(path)).astype(np.float32) * 0.001

def world_points(cam, depth):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    x = (u - cam.cx) * depth / cam.fx
    y = -(v - cam.cy) * depth / cam.fy
    z = -depth
    P = np.stack([x, y, z], -1) @ cam.R.T + cam.t
    return P

def target_mask(rgb):
    f = rgb.astype(np.float32) / 255
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    purple = (b > g + 0.10) & (r > g + 0.03) & (b > 0.3)
    white = (b > 0.55) & (g > 0.5) & (r > 0.5) & (b > 0.8 * r)
    bluecube = (b > r + 0.25) & (g < 0.35)
    return (purple | white) & ~bluecube

def cube_mask(rgb, col):
    a = rgb.astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if col == 'red': return (r > 90) & (g < 0.3 * r) & (b < 0.3 * r)
    if col == 'green': return (g > 90) & (r < 0.5 * g) & (b < 0.5 * g)
    if col == 'blue': return (b > 90) & (r < 0.4 * b) & (g < 0.45 * b)
    raise ValueError(col)

def detect_targets(rgb, P=None, cam=None, zmax=0.02, min_px=12):
    """Return list of (xy world [m], npix)."""
    mk = target_mask(rgb)
    if P is not None:
        z = P[..., 2]
        mk &= (z < zmax) & (z > -0.01)
        # exclude the static button region handled by caller
    lab, n = ndimage.label(ndimage.binary_dilation(mk, iterations=1))
    out = []
    for k in range(1, n + 1):
        sel = (lab == k) & mk
        cnt = int(sel.sum())
        if cnt < min_px:
            continue
        f = rgb[sel].astype(np.float32) / 255
        ring = ((f[:, 0] - f[:, 1] > 0.08) & (f[:, 2] - f[:, 1] > 0.03)).mean()
        if ring < 0.12:
            continue
        if P is not None:
            xy = np.median(P[sel][:, :2], axis=0)
        else:
            ys, xs = np.nonzero(sel)
            xy = cam.to_plane(xs.mean(), ys.mean(), 0.009)[:2]
        if xy[0] < -0.45:
            continue
        out.append((np.array(xy), cnt))
    return out

def wrist_color(rgb):
    """Dominant cube color in the central-lower wrist region (cube between fingers)."""
    h, w = rgb.shape[:2]
    reg = rgb[h // 3:, w // 5: 4 * w // 5]
    counts = {c: int(cube_mask(reg, c).sum()) for c in COLORS}
    best = max(counts, key=counts.get)
    return best, counts

def parse_instruction(text):
    t = text.lower()
    col = next((c for c in COLORS if c in t), None)
    ords = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6, 'seventh': 7, 'eighth': 8,
            '1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5, 'last': -1}
    n = None
    for k, v in ords.items():
        if re.search(r'\b' + k + r'\b', t):
            n = v
            break
    return col, n
