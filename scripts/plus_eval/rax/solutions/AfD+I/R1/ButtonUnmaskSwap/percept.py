"""Perception: calibration check, point clouds, scene parts, bin tracker."""
import math
import numpy as np

from geom import Camera, CONVENTIONS, cloud
from obsio import log, parse_points

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None


# ------------------------------------------------------------------ colour
def hsv(rgb):
    """rgb uint8 (...,3) -> hue deg [0,360), sat [0,1], val [0,1]"""
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    mn = a.min(-1)
    df = mx - mn
    s = np.where(mx > 1e-6, df / np.maximum(mx, 1e-6), 0)
    h = np.zeros_like(mx)
    dfs = np.where(df > 1e-6, df, 1)
    hr = ((g - b) / dfs) % 6
    hg = (b - r) / dfs + 2
    hb = (r - g) / dfs + 4
    h = np.where(mx == r, hr, np.where(mx == g, hg, hb)) * 60.0
    h = np.where(df > 1e-6, h, 0)
    return h, s, mx


def color_mask(h, s, v, name):
    if name == 'red':
        return (s > 0.5) & (v > 0.25) & ((h < 8) | (h > 340))
    if name == 'green':
        return (s > 0.45) & (v > 0.2) & (h > 85) & (h < 165)
    if name == 'blue':
        return (s > 0.45) & (v > 0.2) & (h > 200) & (h < 262)
    if name == 'yellow':
        return (s > 0.5) & (v > 0.4) & (h > 46) & (h < 70)
    if name == 'purple':
        return (s > 0.3) & (v > 0.25) & (h > 265) & (h < 320)
    if name == 'pink':
        return (s > 0.3) & (v > 0.4) & (h >= 320) & (h <= 340)
    if name == 'cyan':
        return (s > 0.4) & (v > 0.3) & (h >= 165) & (h <= 200)
    if name == 'orange':
        return (s > 0.75) & (v > 0.5) & (h > 22) & (h < 42)
    if name == 'black':
        return (v < 0.18)
    if name == 'white':
        return (s < 0.12) & (v > 0.8)
    if name in ('gray', 'brown'):
        return np.zeros_like(s, bool)
    return np.zeros_like(s, bool)


# ------------------------------------------------------------------ grid helpers
def label2d(mask):
    if _ndi is not None:
        lab, n = _ndi.label(mask)
        return lab, n
    # tiny BFS fallback
    H, W = mask.shape
    lab = np.zeros((H, W), int)
    n = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not lab[i, j]:
                n += 1
                st = [(i, j)]
                lab[i, j] = n
                while st:
                    a, b = st.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < H and 0 <= y < W and mask[x, y] and not lab[x, y]:
                            lab[x, y] = n
                            st.append((x, y))
    return lab, n


def dilate(mask, it=1):
    if _ndi is not None:
        return _ndi.binary_dilation(mask, iterations=it)
    m = mask.copy()
    for _ in range(it):
        m2 = m.copy()
        m2[1:] |= m[:-1]; m2[:-1] |= m[1:]; m2[:, 1:] |= m[:, :-1]; m2[:, :-1] |= m[:, 1:]
        m = m2
    return m


class Grid:
    """XY occupancy grid over a fixed window"""

    def __init__(self, xmin, xmax, ymin, ymax, cell=0.01):
        self.x0, self.y0, self.cell = xmin, ymin, cell
        self.nx = max(1, int(math.ceil((xmax - xmin) / cell)))
        self.ny = max(1, int(math.ceil((ymax - ymin) / cell)))

    def idx(self, xy):
        i = np.floor((xy[:, 0] - self.x0) / self.cell).astype(int)
        j = np.floor((xy[:, 1] - self.y0) / self.cell).astype(int)
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        return i, j, ok

    def mask(self, xy):
        m = np.zeros((self.nx, self.ny), bool)
        i, j, ok = self.idx(xy)
        m[i[ok], j[ok]] = True
        return m

    def lookup(self, m, xy):
        i, j, ok = self.idx(xy)
        out = np.zeros(len(xy), bool)
        out[ok] = m[i[ok], j[ok]]
        return out

    def centre(self, i, j):
        return np.stack([self.x0 + (np.asarray(i) + 0.5) * self.cell, self.y0 + (np.asarray(j) + 0.5) * self.cell], -1)


# ------------------------------------------------------------------ calibration
class Calib:
    """Chooses, per camera, the back-projection convention that matches the
    official robomme_pixel_to_position tool."""

    def __init__(self):
        self.conv = {}
        self.err = {}
        self.depth_swap = False
        self.ok = {}

    def camera(self, reply, cam):
        K = reply.Ks.get(cam)
        P = reply.poses.get(cam)
        if K is None or P is None:
            return None
        conv = self.conv.get(cam, ('gl', False, 0.5))
        return Camera(K, P[0], P[1], conv)

    def depth(self, reply, cam):
        if self.depth_swap:
            cam = 'wrist' if cam == 'front' else 'front'
        return reply.images.get((cam, 'depth'))

    def calibrate(self, ctx, reply):
        """call pixel_to_position on sample pixels; choose convention."""
        sels = []
        for cam in ('front', 'wrist'):
            d = reply.images.get((cam, 'depth'))
            H, W = (256, 256) if d is None else d.shape
            for (fu, fv) in ((0.5, 0.5), (0.3, 0.6), (0.7, 0.6), (0.5, 0.8), (0.3, 0.35), (0.7, 0.35), (0.5, 0.25), (0.2, 0.85)):
                sels.append({'camera': cam, 'u': float(int(fu * W)), 'v': float(int(fv * H))})
        try:
            res = ctx.call('robomme_pixel_to_position', observation_step_id=int(reply.step or 0),
                           selections=sels, depth_window_radius=0)
            pts = parse_points(res)
        except Exception as e:
            log('calib: pixel_to_position failed', e)
            return False
        if len(pts) != len(sels):
            log('calib: unexpected point count', len(pts), len(sels))
        best_any = False
        for swap in (False, True):
            results = {}
            for cam in ('front', 'wrist'):
                dcam = ('wrist' if cam == 'front' else 'front') if swap else cam
                d = reply.images.get((dcam, 'depth'))
                K = reply.Ks.get(cam)
                P = reply.poses.get(cam)
                if d is None or K is None or P is None:
                    continue
                best = (1e9, None)
                for conv in CONVENTIONS:
                    c = Camera(K, P[0], P[1], conv)
                    errs = []
                    for s, p in zip(sels, pts):
                        if s['camera'] != cam or p is None:
                            continue
                        u, v = int(s['u']), int(s['v'])
                        if not (0 <= v < d.shape[0] and 0 <= u < d.shape[1]) or d[v, u] <= 0:
                            continue
                        q = c.backproject(np.array([u]), np.array([v]), np.array([d[v, u]]))[0]
                        errs.append(float(np.linalg.norm(q - p)))
                    if len(errs) >= 3:
                        e = float(np.median(errs))
                        if e < best[0]:
                            best = (e, conv)
                results[cam] = best
            if results and all(v[0] < 0.02 for v in results.values()):
                for cam, (e, conv) in results.items():
                    self.conv[cam] = conv
                    self.err[cam] = e
                    self.ok[cam] = True
                self.depth_swap = swap
                log('calib: ok swap=%s' % swap, {c: (round(e, 4), conv) for c, (e, conv) in results.items()})
                return True
            log('calib attempt swap=%s' % swap, {c: (round(e, 4), conv) for c, (e, conv) in results.items()})
            # partial success
            for cam, (e, conv) in results.items():
                if e < 0.02 and not swap:
                    self.conv[cam] = conv
                    self.err[cam] = e
                    self.ok[cam] = True
                    best_any = True
        return best_any


# ------------------------------------------------------------------ point clouds
class CamPoints:
    def __init__(self, cam, P, h, s, v, u, vv, dist):
        self.cam, self.P, self.h, self.s, self.v, self.u, self.vv, self.dist = cam, P, h, s, v, u, vv, dist


def cam_points(reply, calib, cam, step=2):
    c = calib.camera(reply, cam)
    d = calib.depth(reply, cam)
    rgb = reply.images.get((cam, 'rgb'))
    if c is None or d is None:
        return None
    if cam in calib.ok and not calib.ok[cam]:
        return None
    H, W = d.shape
    vs, us = np.mgrid[0:H:step, 0:W:step]
    dd = d[::step, ::step]
    ok = (dd > 1e-4) & (dd < 5.0)
    us, vs, dd = us[ok], vs[ok], dd[ok]
    P = c.backproject(us, vs, dd)
    if rgb is not None and rgb.shape[:2] == d.shape:
        col = rgb[vs, us]
    else:
        col = np.full((len(us), 3), 128, np.uint8)
    h, s, v = hsv(col)
    return CamPoints(cam, P, h, s, v, us, vs, dd)


def table_height(cps):
    zs = []
    for cp in cps:
        if cp is None:
            continue
        m = (cp.h > 4) & (cp.h < 45) & (cp.s > 0.3) & (cp.v > 0.2)
        if m.sum() > 50:
            zs.append(cp.P[m, 2])
    if not zs:
        return None
    z = np.concatenate(zs)
    # mode of histogram (table is the dominant plane)
    hist, edges = np.histogram(z, bins=np.arange(z.min() - 0.005, z.max() + 0.01, 0.004))
    k = int(np.argmax(hist))
    zc = 0.5 * (edges[k] + edges[k + 1])
    near = z[np.abs(z - zc) < 0.006]
    return float(np.median(near)) if len(near) else float(zc)


# ------------------------------------------------------------------ tracker
VMAX_STEP = 0.012  # m/step cap on track velocity (demo swaps ~1.5-3 mm/step)
class Tracker:
    def __init__(self, seeds, assign_r=0.04, min_pts=6):
        self.p = np.array(seeds, float).reshape(-1, 2)
        self.v = np.zeros_like(self.p)
        self.last = np.zeros(len(self.p), int)
        self.seen = np.zeros(len(self.p), int)
        self.assign_r = assign_r
        self.min_pts = min_pts
        self.step = None
        self.close = {}  # (i,j) -> count of close encounters
        self.moving = []  # history of max speed per update
        self.frozen = set()  # tracks not to update (held)
        self.hist = [[] for _ in range(len(self.p))]
        self.merge_d = None
        self.merged_steps = 0
        self.pair = {}  # j -> (partner, midpoint, last confirmed step, n confirmations)
        self.pair_used = 0

    def predict(self, step):
        if self.step is None:
            return self.p.copy()
        dt = max(0, step - self.step)
        return self.p + self.v * min(dt, 10)

    def update(self, step, xy, close_d=0.05):
        if self.step is not None and step <= self.step and xy is None:
            return
        dt = 1 if self.step is None else max(1, step - self.step)
        pred = self.predict(step)
        newp = pred.copy()
        counts = np.zeros(len(pred), int)
        merged = set()
        if xy is not None and len(xy):
            C = pred.copy()
            for it in range(3):
                d2 = ((xy[:, None, :] - C[None]) ** 2).sum(-1)
                a = d2.argmin(1)
                dm = d2[np.arange(len(xy)), a]
                for j in range(len(C)):
                    if j in self.frozen:
                        continue
                    sel = (a == j) & (dm < self.assign_r ** 2)
                    counts[j] = int(sel.sum())
                    if counts[j] >= self.min_pts:
                        C[j] = np.median(xy[sel], 0)
                    else:
                        C[j] = pred[j]
            newp = C
            # coast through merges: two tracks closer than merge_d cannot be separated from points;
            # keep their predicted motion, shifted so the pair's mean matches the blob centroid.
            if self.merge_d:
                n = len(newp)
                D = np.linalg.norm(newp[:, None] - newp[None], axis=-1)
                done = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        if i in done or j in done or i in self.frozen or j in self.frozen:
                            continue
                        if D[i, j] < self.merge_d or np.linalg.norm(pred[i] - pred[j]) < self.merge_d:
                            mid = 0.5 * (pred[i] + pred[j])
                            d2b = np.linalg.norm(xy - mid, axis=1)
                            sel = d2b < 0.5 * self.merge_d + self.assign_r
                            if sel.sum() >= self.min_pts:
                                shift = np.median(xy[sel], 0) - mid
                                sh = np.linalg.norm(shift)
                                if sh > 0.01:
                                    shift = shift / sh * 0.01
                            else:
                                shift = np.zeros(2)
                            newp[i] = pred[i] + shift
                            newp[j] = pred[j] + shift
                            merged.update((i, j))
                            done.update((i, j))
        for j in range(len(newp)):
            if j in self.frozen:
                newp[j] = self.p[j]
                self.v[j] = 0
                continue
            if j in merged:
                self.last[j] = step  # velocity kept (coasting)
                continue
            if counts[j] >= self.min_pts:
                vel = (newp[j] - self.p[j]) / dt
                sp = np.linalg.norm(vel)
                if sp > VMAX_STEP:  # implausibly fast
                    vel = vel / sp * VMAX_STEP
                self.v[j] = 0.5 * self.v[j] + 0.5 * vel
                self.last[j] = step
                self.seen[j] += 1
            else:
                self.v[j] *= 0.7
        try:
            self._pairs(step, newp, counts, merged)
        except Exception as e:  # never let the pair model break tracking
            log('pair model error', e)
        if merged:
            self.merged_steps += 1
        self.last_merged = merged
        speed = np.linalg.norm(newp - self.p, axis=1) / dt
        self.p = newp
        self.step = step
        self.moving.append((step, float(speed.max()) if len(speed) else 0.0, int((counts >= self.min_pts).sum())))
        for j in range(len(self.p)):
            self.hist[j].append((step, float(self.p[j][0]), float(self.p[j][1]), int(counts[j])))
            if len(self.hist[j]) > 2000:
                self.hist[j] = self.hist[j][-2000:]
        n = len(self.p)
        if n > 1:
            D = np.linalg.norm(self.p[:, None] - self.p[None], axis=-1)
            for i in range(n):
                for j in range(i + 1, n):
                    if D[i, j] < close_d:
                        self.close[(i, j)] = self.close.get((i, j), 0) + 1
        return counts

    def _pairs(self, step, newp, counts, merged):
        """Swap-pair model: two bins moving with opposite velocities about a fixed midpoint are a
        swapping pair; while one is hidden its position is the reflection of the visible partner
        through the midpoint."""
        n = len(newp)
        vis = np.array([counts[j] >= self.min_pts and j not in merged and j not in self.frozen for j in range(n)])
        sp = np.linalg.norm(self.v, axis=1)
        # detect / refresh pairs among visible moving tracks
        for i in range(n):
            if not vis[i] or sp[i] < 0.0008:
                continue
            for k in range(i + 1, n):
                if not vis[k] or sp[k] < 0.0008:
                    continue
                ratio = sp[i] / sp[k]
                cosv = float(self.v[i] @ self.v[k] / (sp[i] * sp[k]))
                if cosv > -0.7 or not (0.5 < ratio < 2.0) or np.linalg.norm(newp[i] - newp[k]) > 0.25:
                    continue
                mid = 0.5 * (newp[i] + newp[k])
                old = self.pair.get(i)
                if old is not None and old[0] == k:
                    if np.linalg.norm(old[1] - mid) < 0.015:
                        m2 = 0.8 * old[1] + 0.2 * mid
                        self.pair[i] = (k, m2, step, old[3] + 1)
                        self.pair[k] = (i, m2, step, old[3] + 1)
                    continue
                if (i in self.pair and step - self.pair[i][2] < 10) or (k in self.pair and step - self.pair[k][2] < 10):
                    continue
                self.pair[i] = (k, mid, step, 1)
                self.pair[k] = (i, mid, step, 1)
        # use pairs for hidden members
        for i in range(n):
            if i in self.frozen or vis[i] or i in merged:
                continue
            pr = self.pair.get(i)
            if pr is None:
                continue
            k, mid, last, nconf = pr
            if step - last > 80 or nconf < 2 or not vis[k] or self.pair.get(k, (None,))[0] != i:
                continue
            newp[i] = 2 * mid - newp[k]
            self.v[i] = -self.v[k]
            self.p[i] = newp[i]
            self.pair_used += 1
        # expire pairs whose members are both visible and static
        for i in list(self.pair.keys()):
            k, mid, last, nconf = self.pair[i]
            if step - last > 80 or (vis[i] and sp[i] < 0.0004 and step - last > 5):
                self.pair.pop(i, None)

    def static_for(self, thr=0.0015):
        """number of most recent steps during which all tracks moved < thr m/step"""
        k = 0
        last_step = None
        for st, sp, nseen in reversed(self.moving):
            if sp > thr:
                break
            if last_step is not None:
                k += last_step - st
            last_step = st
        return k

    def partner(self, i, exclude=()):
        best, bj = 0, None
        for (a, b), c in self.close.items():
            j = b if a == i else a if b == i else None
            if j is None or j in exclude:
                continue
            if c > best:
                best, bj = c, j
        return bj
