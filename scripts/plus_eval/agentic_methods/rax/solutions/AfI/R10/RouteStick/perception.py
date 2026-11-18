"""Target-disc detection and video->live route mapping."""
import json, math
import numpy as np
from PIL import Image


def qmat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def backproj(u, v, d, K, pose):
    fx, cx, fy, cy = K[0][0], K[0][2], K[1][1], K[1][2]
    pc = np.array([(u-cx)*d/fx, -(v-cy)*d/fy, -d])
    return qmat(pose['quaternion_wxyz']) @ pc + np.array(pose['position_m'])


def components(mask):
    """4-connected components of a boolean mask -> list of (ys, xs)."""
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    comps = []
    ys0, xs0 = np.nonzero(mask)
    for y0, x0 in zip(ys0, xs0):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]; seen[y0, x0] = True; cy, cx = [], []
        while stack:
            y, x = stack.pop(); cy.append(y); cx.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; stack.append((yy, xx))
        comps.append((np.array(cy), np.array(cx)))
    return comps


def find_targets(rgb_path, depth_path, K, P, zmax=0.045):
    rgb = np.array(Image.open(rgb_path).convert('RGB')).astype(int)
    dep = np.array(Image.open(depth_path)).astype(float) * 0.001
    mx = rgb.max(2); mn = rgb.min(2)
    mask = (mn > 150) & (mx - mn < 40)
    mask[:25] = False
    out = []
    for ys, xs in components(mask):
        if len(xs) < 6 or len(xs) > 400:
            continue
        d = dep[ys, xs]; d = d[d > 0]
        if len(d) == 0:
            continue
        w = backproj(xs.mean(), ys.mean(), float(np.median(d)), K, P)
        if w[2] > zmax:
            continue
        out.append({'px': (float(xs.mean()), float(ys.mean())), 'n': int(len(xs)), 'xyz': [float(c) for c in w]})
    out.sort(key=lambda t: t['xyz'][1])
    return out


def frame_targets(frame, ws):
    files = {(f['camera'], f['kind']): f['path'] for f in frame['files']}
    K = frame['camera_intrinsics']['front']['matrix']; P = frame['camera_poses']['front']
    return find_targets(ws / files[('front', 'rgb')], ws / files[('front', 'depth')], K, P)


class RouteMapper:
    """Map video-frame points to the live layout segment by segment along the target line."""

    def __init__(self, src, dst):
        self.src = [np.array(p[:2]) for p in src]
        self.dst = [np.array(p[:2]) for p in dst]

    def map(self, p):
        q = np.array(p[:2], float)
        n = len(self.src)
        if n == 1:
            r = self.dst[0] + (q - self.src[0])
            return [float(r[0]), float(r[1]), p[2]]
        best = None
        for k in range(n - 1):
            a, b = self.src[k], self.src[k+1]
            ab = b - a; L = np.linalg.norm(ab); u = ab / L
            t = float(np.dot(q - a, u)) / L
            tc = min(1.0, max(0.0, t))
            dist = np.linalg.norm(q - (a + tc * ab))
            if best is None or dist < best[0] - 1e-9:
                best = (dist, k, t, float(np.cross(u, q - a)))
        _, k, t, lat = best
        a2, b2 = self.dst[k], self.dst[k+1]
        ab2 = b2 - a2; u2 = ab2 / np.linalg.norm(ab2); nrm = np.array([-u2[1], u2[0]])
        r = a2 + t * ab2 + lat * nrm
        return [float(r[0]), float(r[1]), p[2]]


def line_inliers(targets, tol=0.035):
    """Keep the largest collinear subset (targets lie on one line on the table)."""
    pts = [np.array(t['xyz'][:2]) for t in targets]
    n = len(pts)
    if n <= 2:
        return targets
    best = []
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[j] - pts[i]; L = np.linalg.norm(d)
            if L < 1e-6:
                continue
            u = d / L
            inl = [k for k in range(n) if abs(float(np.cross(u, pts[k] - pts[i]))) < tol]
            if len(inl) > len(best):
                best = inl
    return [targets[k] for k in best]


def order_along_line(targets):
    pts = np.array([t['xyz'][:2] for t in targets])
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    u = vt[0]
    if u[1] < 0:
        u = -u
    s = (pts - c) @ u
    return [targets[k] for k in np.argsort(s)]


def stick_tip_offset(frame, ws, max_depth=0.3):
    """World offset (tip - EE position) of the drawing stick, from the wrist depth.

    The stick is the only thing within max_depth of the wrist camera; its tip is
    the stick point lowest in world z (the stick hangs below the hand).
    """
    files = {(f['camera'], f['kind']): f['path'] for f in frame['files']}
    K = frame['camera_intrinsics']['wrist']['matrix']; P = frame['camera_poses']['wrist']
    d = np.array(Image.open(ws / files[('wrist', 'depth')])).astype(float) * 0.001
    m = (d > 0) & (d < max_depth)
    ys, xs = np.nonzero(m)
    if len(xs) < 30:
        return None
    pts = np.array([backproj(x, y, d[y, x], K, P) for y, x in zip(ys, xs)])
    zs = pts[:, 2]
    low = pts[zs <= np.min(zs) + 0.003]
    tip = low.mean(0)
    ee = np.array(frame['state'][:3])
    return [float(c) for c in (tip - ee)], int(len(xs))
