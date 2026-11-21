"""Conditioning-video analysis for VideoPlaceOrder: track targets (shell-game swaps)
and the named cube's placement events, in front-camera pixel space."""
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

ORD = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6,
       'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
       '1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5, '6th': 6}


def parse_instruction(text):
    t = text.lower()
    color = None
    for c in ('red', 'green', 'blue', 'yellow'):
        if c + ' cube' in t or c in t.split():
            color = c
            break
    k = None
    for w, n in ORD.items():
        if (' ' + w + ' ') in (' ' + t.replace(',', ' ') + ' '):
            k = n
            break
    last = (' last ' in ' ' + t + ' ')
    return color, k, last


MASK_MODE = 'm1'
MIN_PX = 8


def target_mask_m1(a):
    a = a.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (b > r + 8) & (b > g + 15) & (b > 100) & (r > 60) & (g > 35)


def target_mask(a):
    if MASK_MODE == 'm1':
        return target_mask_m1(a)
    ai = a.astype(int)
    r, g, b = ai[..., 0], ai[..., 1], ai[..., 2]
    mn = np.minimum(np.minimum(r, g), b)
    purple = (b > 0.6 * r) & (b > g + 12) & (r > g + 12) & (b > 70)
    white = (mn > 150) & (b > 0.75 * r) & (b >= g - 35) & (r - g > 12)
    white2 = (b > g + 25) & (b > r + 10) & (r > 110)
    mk = purple | white | white2
    cm = cube_mask(a, 'red') | cube_mask(a, 'green') | cube_mask(a, 'blue')
    cm = ndimage.binary_dilation(cm, iterations=3)
    return mk & ~cm


def cube_mask(a, color):
    a = a.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if color == 'red':
        return (r > 140) & (g < 70) & (b < 70)
    if color == 'green':
        return (g > 120) & (r < 90) & (b < 90)
    if color == 'blue':
        return (b > 110) & (r < 60) & (g < 60)
    if color == 'yellow':
        return (r > 170) & (g > 170) & (b < 80)
    raise ValueError(color)


def blobs(mask, min_px=None, close=2):
    if min_px is None:
        min_px = MIN_PX
    if close:
        mask = ndimage.binary_closing(mask, iterations=close)
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) >= min_px:
            out.append((xs.mean(), ys.mean(), len(xs)))
    return out


def cube_centroid(a, color, min_px=15):
    bl = blobs(cube_mask(a, color), min_px=min_px, close=1)
    if not bl:
        return None
    x, y, n = max(bl, key=lambda b: b[2])
    return (x, y, n)


DEPTH_MODE = False
DEPTH_DIL = 2
Z_LO, Z_HI = 0.0045, 0.0125


def target_mask_depth(z):
    """Targets are ~9 mm-high discs on the table: a world-z band, minus anything near taller objects."""
    zz = np.nan_to_num(z, nan=-1.0)
    band = (zz > Z_LO) & (zz < Z_HI)
    if DEPTH_DIL <= 0:
        return band
    high = ndimage.binary_dilation(zz >= Z_HI, iterations=DEPTH_DIL)
    return band & ~high


def detect_targets(a, n_expected=None, z=None):
    """Depth band when a world-z map is available (lighting invariant); else m1 colour mask,
    under coloured lighting falling back to the m5 mask when it explains more targets."""
    global MASK_MODE
    if z is not None and DEPTH_MODE:
        return blobs(target_mask_depth(z), min_px=20)
    b1 = blobs(target_mask_m1(a), min_px=8)
    if n_expected is None or not ADAPTIVE:
        return b1
    if len(b1) >= n_expected:
        return b1
    save = MASK_MODE
    MASK_MODE = 'm5'
    try:
        b5 = blobs(target_mask(a), min_px=30)
    finally:
        MASK_MODE = save
    if abs(len(b5) - n_expected) < abs(len(b1) - n_expected):
        return b5
    return b1


ADAPTIVE = True
GATE0 = 12.0
MERGE_RATIO = 1.4


class TargetTracker:
    """Multi-target tracker with constant-velocity prediction, merge handling and slot snapping."""
    def __init__(self, dets, t0):
        self.x = np.array([[d[0], d[1]] for d in dets], float)
        self.area = np.array([d[2] for d in dets], float)
        self.slots = self.x.copy()
        self.v = np.zeros_like(self.x)
        self.t = t0
        self.n = len(dets)
        self.history = [(t0, self.x.copy())]

    def update(self, dets, t):
        """Area-aware assignment: a blob can absorb k targets (k ~ area / slot area)."""
        dt = max(t - self.t, 1e-3)
        if not hasattr(self, 'moving'):
            self.moving = np.zeros(self.n, bool)
        pred = self.x + self.v * dt
        D = np.array([[d[0], d[1]] for d in dets], float) if dets else np.zeros((0, 2))
        A = np.array([d[2] for d in dets], float)
        newx = self.x.copy(); newv = self.v * 0.8
        if len(D) == 0:
            self.t = t; self.history.append((t, self.x.copy())); return
        # expected single-target area near each detection
        near = np.argmin(np.linalg.norm(D[:, None] - self.slots[None], axis=2), axis=1)
        exp_a = self.area[near]
        copies = np.clip(np.round(A / exp_a), 1, 3).astype(int)
        cols = []
        for j in range(len(D)):
            if A[j] < 0.3 * exp_a[j]:
                continue  # fragment
            cols += [j] * copies[j]
        if not cols:
            self.t = t; self.history.append((t, self.x.copy())); return
        cols = np.array(cols)
        speed = np.linalg.norm(self.v, axis=1) * dt
        gate = np.clip(GATE0 + 1.0 * speed, GATE0, 35.0)
        C = np.linalg.norm(pred[:, None] - D[cols][None], axis=2)
        # small-but-not-fragment blobs may not pull static targets
        for r in range(self.n):
            if not self.moving[r]:
                k = int(np.argmin(np.linalg.norm(self.slots - self.x[r], axis=1)))
                small = A[cols] < 0.6 * self.area[k]
                C[r, small & (np.linalg.norm(D[cols] - self.x[r], axis=1) > 3.0)] = 1e6
        ri, ci = linear_sum_assignment(C)
        match = {}
        for r, c in zip(ri, ci):
            if C[r, c] < gate[r]:
                match[r] = cols[c]
        owner = {}
        for r, j in match.items():
            owner.setdefault(j, []).append(r)
        for j, rs in owner.items():
            z = D[j]
            if len(rs) == 1 and not self.moving[rs[0]]:
                r = rs[0]
                ks = int(np.argmin(np.linalg.norm(self.slots - self.x[r], axis=1)))
                if A[j] > MERGE_RATIO * self.area[ks]:
                    # oversized blob on a static target: a mover is passing over it
                    newx[r] = self.x[r]; newv[r] = np.zeros(2)
                    lost = [q for q in range(self.n) if q not in match and self.moving[q]
                            and np.linalg.norm(pred[q] - z) < 35]
                    if lost:
                        q = min(lost, key=lambda q: np.linalg.norm(pred[q] - z))
                        f = min(self.area[ks] / max(A[j], 1.0), 0.75)
                        est = z + (z - self.x[r]) * f / (1 - f)
                        newv[q] = 0.5 * self.v[q] + 0.5 * (est - self.x[q]) / dt
                        newx[q] = est
                        match[q] = j
                    continue
            if len(rs) == 1:
                r = rs[0]
                vel = (z - self.x[r]) / dt
                newv[r] = 0.4 * self.v[r] + 0.6 * vel
                newx[r] = z
                continue
            # shared blob: the track(s) whose slot is closest to the centroid stays, others unmix
            dist_slot = []
            for r in rs:
                ds = np.linalg.norm(self.slots - self.x[r], axis=1)
                dist_slot.append(ds.min() if not self.moving[r] else 1e3)
            order = np.argsort([np.linalg.norm(z - self.x[r]) if not self.moving[r] else 1e3 for r in rs])
            stay = rs[order[0]] if not self.moving[rs[order[0]]] else None
            if stay is not None and np.linalg.norm(z - self.x[stay]) > 12:
                stay = None
            movers = [r for r in rs if r != stay]
            if stay is not None:
                newx[stay] = self.x[stay]; newv[stay] = np.zeros(2)
                ks = int(np.argmin(np.linalg.norm(self.slots - self.x[stay], axis=1)))
                f = min(self.area[ks] / max(A[j], 1.0), 0.75)
                if len(movers) == 1:
                    r = movers[0]
                    est = z + (z - self.x[stay]) * f / (1 - f)
                    if np.linalg.norm(est - pred[r]) > 20:
                        est = pred[r]
                    newv[r] = 0.5 * self.v[r] + 0.5 * (est - self.x[r]) / dt
                    newx[r] = est
                    self.moving[r] = True
                    continue
            for r in movers:
                # several movers in one blob: follow prediction, but stay near the blob
                p = pred[r]
                off = p - z
                lim = 0.5 * np.sqrt(A[j] / np.pi)
                if np.linalg.norm(off) > lim:
                    p = z + off / np.linalg.norm(off) * lim
                newx[r] = p
                self.moving[r] = True
        # re-acquisition: full-size unclaimed blobs away from static targets go to lost movers
        unclaimed = [j for j in range(len(D)) if j not in owner and A[j] >= 0.6 * exp_a[j]]
        lost = [r for r in range(self.n) if r not in match and self.moving[r]]
        if unclaimed and lost:
            static_pos = np.array([self.x[r] for r in range(self.n) if not self.moving[r]] or [[1e4, 1e4]])
            unclaimed = [j for j in unclaimed if np.linalg.norm(static_pos - D[j], axis=1).min() > 8]
            if unclaimed:
                CC = np.linalg.norm(pred[lost][:, None] - D[unclaimed][None], axis=2)
                ri, ci = linear_sum_assignment(CC)
                for r_, c_ in zip(ri, ci):
                    if CC[r_, c_] < 50:
                        r = lost[r_]; j = unclaimed[c_]
                        match[r] = j
                        newv[r] = (D[j] - self.x[r]) / dt * 0.5
                        newx[r] = D[j]
        for r in range(self.n):
            if r not in match:
                newx[r] = self.x[r] + self.v[r] * dt * (1.0 if self.moving[r] else 0.0)
        # motion state and snapping
        for r in range(self.n):
            ds = np.linalg.norm(self.slots - newx[r], axis=1)
            k = int(np.argmin(ds))
            sp = np.linalg.norm(newv[r]) * dt
            if ds[k] > 3.0:
                self.moving[r] = True
            if ds[k] < 3.0 and sp < 1.5:
                newx[r] = self.slots[k]
                if self.moving[r] and sp < 0.8:
                    self.moving[r] = False
                    newv[r] = np.zeros(2)
        self.x, self.v, self.t = newx, newv, t
        self.history.append((t, self.x.copy()))

    def slot_assignment(self, tol=6.0):
        D = np.linalg.norm(self.x[:, None] - self.slots[None], axis=2)
        return tuple(int(np.argmin(D[r])) if D[r].min() < tol else -1 for r in range(self.n))


def _z(fr):
    return fr[2] if len(fr) > 2 else None


def analyze(frames, color, k=None, log=print, rest_eps=1.2, rest_min_s=0.5, on_tgt_px=12.0):
    """frames: list of (t_seconds, rgb_front uint8 HxWx3).
    Returns placements with the tracked target identity; identity i starts at slot i
    (initial layout), which is what the execution scene shows."""
    nb = [len(detect_targets(frames[i][1], z=_z(frames[i]))) for i in range(min(10, len(frames)))]
    i0 = int(np.argmax(nb))
    tr = TargetTracker(detect_targets(frames[i0][1], z=_z(frames[i0])), frames[i0][0])
    log(f'targets init frame {i0}: n={tr.n} pos={np.round(tr.x,1).tolist()}')
    ref = [cube_centroid(frames[i][1], color) for i in range(min(10, len(frames)))]
    ref = [c[2] for c in ref if c is not None]
    cmin = max(15, int(0.4 * np.median(ref))) if ref else 15
    log(f'cube ref size {np.median(ref) if ref else None} min_px {cmin}')
    cubes = []
    for i, fr in enumerate(frames):
        t, a = fr[0], fr[1]
        if i > i0:
            tr.update(detect_targets(a, tr.n, z=_z(fr)), t)
        c = cube_centroid(a, color, min_px=cmin)
        cubes.append((t, c, tr.x.copy()))
    segs = []
    cur = None
    prev = None
    for (t, c, X) in cubes:
        if c is None:
            continue
        if prev is not None and np.hypot(c[0] - prev[1][0], c[1] - prev[1][1]) <= rest_eps * max(1.0, (t - prev[0]) / 0.1):
            if cur is None:
                cur = {'t0': prev[0], 'pts': [prev[1][:2]], 'X': [prev[2]]}
            cur['t1'] = t
            cur['pts'].append(c[:2])
            cur['X'].append(X)
        else:
            if cur is not None:
                segs.append(cur)
            cur = None
        prev = (t, c, X)
    if cur is not None:
        segs.append(cur)
    events = []
    for s in segs:
        if s['t1'] - s['t0'] < rest_min_s:
            continue
        p = np.mean(s['pts'], axis=0)
        # identity: vote over the segment
        votes = {}
        dmin = 1e9
        for X in s['X']:
            d = np.linalg.norm(X - (p + np.array([0, 3.0])), axis=1)
            j = int(np.argmin(d))
            dmin = min(dmin, d[j])
            if d[j] < on_tgt_px:
                votes[j] = votes.get(j, 0) + 1
        tgt = max(votes, key=votes.get) if votes else None
        slot = None
        ds = np.linalg.norm(tr.slots - (p + np.array([0, 3.0])), axis=1)
        if ds.min() < on_tgt_px:
            slot = int(np.argmin(ds))
        ev = {'t0': s['t0'], 't1': s['t1'], 'pos': p.tolist(), 'target': tgt, 'slot': slot, 'dist': float(dmin)}
        if events and events[-1]['slot'] == slot and (slot is not None or np.linalg.norm(np.array(events[-1]['pos']) - p) < 6.0):
            events[-1]['t1'] = s['t1']
            if events[-1]['target'] is None:
                events[-1]['target'] = tgt
            continue
        events.append(ev)
    for e in events:
        log(f"rest {e['t0']:.2f}-{e['t1']:.2f}s pos=({e['pos'][0]:.1f},{e['pos'][1]:.1f}) slot={e['slot']} id={e['target']} d={e['dist']:.1f}")
    placements = [e for e in events if e['slot'] is not None]
    return {'tracker': tr, 'events': events, 'placements': placements, 'slots': tr.slots.copy()}
