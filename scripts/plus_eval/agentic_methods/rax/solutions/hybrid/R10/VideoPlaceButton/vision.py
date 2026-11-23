"""Pixel-level perception for VideoPlaceButton (front camera, 256x256 RGB)."""
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

def _rgb(a):
    a = a.astype(np.int32)
    return a[..., 0], a[..., 1], a[..., 2]

def hsv(a):
    a = a.astype(np.float32) / 255.0
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    mx = np.maximum(np.maximum(R, G), B); mn = np.minimum(np.minimum(R, G), B)
    d = mx - mn + 1e-6
    h = np.where(mx == R, ((G - B) / d) % 6, np.where(mx == G, (B - R) / d + 2, (R - G) / d + 4)) * 60.0
    s = (mx - mn) / (mx + 1e-6)
    return h, s, mx

TARGET_MODE = {'mode': 'hue'}

def target_mask(a):
    """Coloured ring parts of the targets: magenta/purple/pink hues (table is orange,
    cubes are saturated primaries). 'wide' fallback: any non-orange, moderately
    saturated colour."""
    h, s, v = hsv(a)
    if TARGET_MODE['mode'] == 'wide':
        return ((h > 55) | (h < 0)) & (s > 0.12) & (s < 0.82) & (v > 0.35)
    return (h >= 255) & (h <= 350) & (s > 0.12) & (v > 0.35)

def cube_mask(a, color):
    R, G, B = _rgb(a)
    if color == 'red':
        return (R > 120) & (G < 70) & (B < 70)
    if color == 'blue':
        return (B > 110) & (R < 70) & (G < 70)
    if color == 'green':
        return (G > 110) & (R < 100) & (B < 100)
    raise ValueError(color)

def white_mask(a):
    R, G, B = _rgb(a)
    return (R > 200) & (G > 200) & (B > 200) & (abs(R - G) < 20) & (abs(G - B) < 20)

def blobs(mask, minarea):
    lab, n = ndimage.label(mask)
    out = []
    if n == 0:
        return out
    idx = np.arange(1, n + 1)
    areas = ndimage.sum(mask, lab, idx)
    coms = ndimage.center_of_mass(mask, lab, idx)
    for a, (cy, cx) in zip(areas, coms):
        if a >= minarea:
            out.append((float(cx), float(cy), float(a)))
    return out

def detect_targets(a, minarea=30):
    t = ndimage.binary_closing(target_mask(a), iterations=1)
    return blobs(t, minarea)

def detect_cube(a, color, minarea=25):
    b = blobs(cube_mask(a, color), minarea)
    if not b:
        return None
    return max(b, key=lambda x: x[2])

def detect_button(a):
    """Largest white blob (button base) -> (cx, cy, area, mask)."""
    wm = white_mask(a)
    lab, n = ndimage.label(wm)
    if n == 0:
        return None
    sizes = ndimage.sum(wm, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    m = lab == k
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean()), float(m.sum()), m)


class TargetTracker:
    """Track identical targets through swaps with constant-velocity prediction
    and Hungarian assignment. Tracks coast when unmatched (occlusion/merge)."""

    def __init__(self, init_dets, gate=28.0):
        self.pos = np.array([[d[0], d[1]] for d in init_dets], float)
        self.vel = np.zeros_like(self.pos)
        self.miss = np.zeros(len(self.pos), int)
        self.gate = gate
        self.history = [self.pos.copy()]

    def update(self, dets, dt=1.0):
        n = len(self.pos)
        pred = self.pos + self.vel * dt
        if dets:
            D = np.array([[d[0], d[1]] for d in dets], float)
            C = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            # also allow matching to stationary position (handles start of motion)
            C2 = np.linalg.norm(self.pos[:, None, :] - D[None, :, :], axis=2)
            C = np.minimum(C, C2 + 2.0)
            r, c = linear_sum_assignment(C)
            matched = set()
            for i, j in zip(r, c):
                if C[i, j] <= self.gate * max(1.0, dt ** 0.5):
                    newp = D[j]
                    v = (newp - self.pos[i]) / dt
                    self.vel[i] = 0.5 * self.vel[i] + 0.5 * v
                    self.pos[i] = newp
                    self.miss[i] = 0
                    matched.add(i)
            for i in range(n):
                if i not in matched:
                    self.miss[i] += 1
                    if self.miss[i] <= 3:
                        self.pos[i] = pred[i]
                    self.vel[i] *= 0.5
        else:
            self.miss += 1
        self.history.append(self.pos.copy())
