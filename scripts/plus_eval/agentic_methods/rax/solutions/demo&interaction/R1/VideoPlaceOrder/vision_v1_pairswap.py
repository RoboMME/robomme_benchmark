"""Video reasoning for VideoPlaceOrder.

Input: ordered conditioning frames (timestamp_s, front RGB uint8 HxWx3, wrist RGB or None).
Output: which target slot (pixel position in the front view) holds the target identity on which
the goal cube was placed for the k-th time.

Model learned from the released demonstrations:
- N identical purple targets rest on a fixed set of slot positions;
- shuffles move targets along arcs between slots (identities are permuted);
- the robot repeatedly places the goal cube on targets (cube rests on a slot, gripper opens);
- placements are counted by target identity; the identity must be followed to the end of the video.
"""
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

ORDINALS = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6,
            'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10, '1st': 1, '2nd': 2, '3rd': 3,
            '4th': 4, '5th': 5, '6th': 6, '7th': 7, '8th': 8}
COLORS = ('red', 'green', 'blue')


def parse_instruction(text):
    t = (text or '').lower().replace(',', ' ').replace('.', ' ')
    words = t.split()
    color = None
    # colour named right before "cube"
    for i, w in enumerate(words):
        if w.startswith('cube') and i > 0 and words[i - 1] in COLORS:
            color = words[i - 1]
            break
    if color is None:
        for w in words:
            if w in COLORS:
                color = w
                break
    k = None
    for w in words:
        if w in ORDINALS:
            k = ORDINALS[w]
            break
    if k is None:
        for w in words:
            if w.isdigit():
                k = int(w)
                break
    return color, k


def target_mask(fr):
    """Saturated purple ring pixels of the targets. Robust to the coloured spot lights seen in the
    demonstrations: B - G stays positive on the purple ring (wood: strongly negative), the ring is
    saturated (arm under lavender light is not), and R is not tiny (blue cube)."""
    fr = fr.astype(np.int16)
    r, g, b = fr[..., 0], fr[..., 1], fr[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    m1 = (b > g + 15) & (mx - mn > 30) & ((mx - mn) * 10 > mx * 3) & (r > 60) & (r * 10 > b * 5)
    # purple ring under a strong warm spot light: blue drops but stays well above the wood's
    m2 = (r > 150) & (g * 100 < r * 72) & (b * 100 > r * 52) & (b > g - 30)
    m = (m1 | m2).astype(np.uint8)
    # JPEG/antialias fringe around the blue cube looks purple: drop pixels next to saturated blue
    blue = ((b > 120) & (r * 10 < b * 4) & (g * 10 < b * 4)).astype(np.uint8)
    if blue.any():
        m[cv2.dilate(blue, np.ones((5, 5), np.uint8)) > 0] = 0
    return m


def target_blobs(fr, min_area=25):
    mk = cv2.morphologyEx(target_mask(fr), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mk)
    return [(float(cen[i][0]), float(cen[i][1]), int(st[i, 4])) for i in range(1, n) if st[i, 4] >= min_area]


def slot_shape(front, slots):
    """Average normalised second-moment matrix (det = 1) of target blobs at the slots: the image of a
    table-plane circle is approximately an ellipse x^T M^-1 x = const."""
    mk = cv2.morphologyEx(target_mask(front), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mk)
    covs = []
    for i in range(1, n):
        if st[i, 4] < 60:
            continue
        if min(np.hypot(cen[i][0] - s[0], cen[i][1] - s[1]) for s in slots) > 5:
            continue
        ys, xs = np.nonzero(lab == i)
        C = np.cov(np.vstack([xs, ys]).astype(float))
        if np.linalg.det(C) <= 0:
            continue
        covs.append(C / np.sqrt(np.linalg.det(C)))
    if not covs:
        return np.diag([1.0 / 0.6, 0.6])
    M = np.mean(covs, axis=0)
    return M / np.sqrt(np.linalg.det(M))


def cube_mask(fr, color):
    fr = fr.astype(np.int16)
    r, g, b = fr[..., 0], fr[..., 1], fr[..., 2]
    if color == 'red':
        m = (r > 90) & (g * 10 < r * 4) & (b * 10 < r * 4)
    elif color == 'green':
        m = (g > 90) & (r * 10 < g * 6) & (b * 10 < g * 6)
    else:
        m = (b > 80) & (r * 10 < b * 5) & (g * 10 < b * 5)
    return m.astype(np.uint8)


def cube_blob(fr, color, min_area=12):
    """Largest blob of the cube colour: (x_mean, y_bottom, area, x_c, y_c) or None."""
    mk = cube_mask(fr, color)
    n, lab, st, cen = cv2.connectedComponentsWithStats(mk)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, 4]))
    if st[i, 4] < min_area:
        return None
    ys, xs = np.nonzero(lab == i)
    return (float(xs.mean()), float(ys.max()), int(st[i, 4]), float(cen[i][0]), float(cen[i][1]))


def _cluster(points, radius):
    """Greedy clustering of (x, y, weight) points; returns centres sorted by support."""
    cl = []
    for x, y, w in points:
        for c in cl:
            if np.hypot(c[0] / c[2] - x, c[1] / c[2] - y) < radius:
                c[0] += x * w; c[1] += y * w; c[2] += w; c[3] += 1
                break
        else:
            cl.append([x * w, y * w, w, 1])
    return [(c[0] / c[2], c[1] / c[2], c[3]) for c in sorted(cl, key=lambda c: -c[3])]


def find_slots(frames, log=print):
    """Slots = target rest positions. Use blobs from frames where blob count is maximal and
    the set is stable; first frames preferred (arm at home, cube not on a target)."""
    cand = []
    for idx in list(range(min(6, len(frames)))) + [len(frames) - 1]:
        bl = [b for b in target_blobs(frames[idx][1]) if b[2] >= 60]
        cand.append((idx, bl))
    best = max(cand, key=lambda c: (len(c[1]), -c[0]))
    slots = [np.array(b[:2]) for b in best[1]]
    # also gather stable blobs across the whole video: points seen at rest in many frames
    pts = []
    step = max(1, len(frames) // 120)
    for idx in range(0, len(frames), step):
        for b in target_blobs(frames[idx][1]):
            if b[2] >= 60:
                pts.append((b[0], b[1], 1.0))
    cl = _cluster(pts, 5.0)
    nsamp = len(range(0, len(frames), step))
    for c in cl:
        if c[2] >= 0.35 * nsamp and min(np.hypot(c[0] - s[0], c[1] - s[1]) for s in slots) > 12:
            log(f'vision: adding stable slot {c[:2]} support {c[2]}/{nsamp}')
            slots.append(np.array(c[:2]))
    return slots


class Tracker:
    def __init__(self, slots, areas=None, log=print):
        self.slots = [s.copy() for s in slots]
        self.pos = [s.copy() for s in slots]
        self.vel = [np.zeros(2) for _ in slots]
        self.coast = [0] * len(slots)
        self.area = list(areas) if areas is not None else [150.0] * len(slots)
        self.log = log

    def update(self, blobs, dt):
        """blobs: list of (x, y, area); dt in source frames (20 Hz)."""
        n = len(self.pos)
        pred = [p + v * dt for p, v in zip(self.pos, self.vel)]
        det = [np.array(b[:2]) for b in blobs]
        dar = [b[2] for b in blobs]
        matched = [None] * n
        used = set()
        if det:
            C = np.array([[np.linalg.norm(p - d) for d in det] for p in pred])
            # stage 1: tight gate
            r, c = linear_sum_assignment(C)
            for i, j in zip(r, c):
                speed = np.linalg.norm(self.vel[i]) * dt
                if C[i, j] < 9.0 + 0.6 * speed + 4.0 * dt ** 0.5:
                    matched[i] = j
                    used.add(j)
            # stage 2: count conservation - orphan detections go to lost tracks (large gate)
            lost = [i for i in range(n) if matched[i] is None]
            orph = [j for j in range(len(det)) if j not in used and dar[j] >= 40]
            if lost and orph:
                C2 = C[np.ix_(lost, orph)]
                r2, c2 = linear_sum_assignment(C2)
                for a, b in zip(r2, c2):
                    i, j = lost[a], orph[b]
                    if C2[a, b] < 70.0:
                        matched[i] = j
                        used.add(j)
        for i in range(n):
            j = matched[i]
            if j is not None:
                d = det[j]
                step = d - self.pos[i]
                newv = step / max(dt, 1e-6)
                if np.linalg.norm(step) < 0.8:
                    newv = np.zeros(2)
                merged_blob = dar[j] > 1.6 * self.area[i]
                if not merged_blob:
                    self.vel[i] = 0.7 * newv + 0.3 * self.vel[i]
                    if np.linalg.norm(self.vel[i]) < 0.15:
                        # at rest: learn area
                        self.area[i] = 0.9 * self.area[i] + 0.1 * dar[j]
                self.pos[i] = d.copy()
                self.coast[i] = 0
            else:
                # lost: merged with another blob or occluded -> coast along velocity for a while
                self.coast[i] += dt
                if self.coast[i] <= 12.0:
                    self.pos[i] = pred[i]
                else:
                    self.vel[i] = np.zeros(2)

    def at_rest(self):
        return all(np.linalg.norm(v) < 0.3 for v in self.vel)

    def near_slots(self, tol=6.0):
        return all(min(np.linalg.norm(p - s) for s in self.slots) < tol for p in self.pos)

    def snap(self):
        """When everything is at rest, make the identity->slot map a permutation."""
        C = np.array([[np.linalg.norm(p - s) for s in self.slots] for p in self.pos])
        r, c = linear_sum_assignment(C)
        for i, j in zip(r, c):
            self.pos[i] = self.slots[j].copy()
            self.vel[i] = np.zeros(2)
            self.coast[i] = 0

    def mapping(self):
        C = np.array([[np.linalg.norm(p - s) for s in self.slots] for p in self.pos])
        r, c = linear_sum_assignment(C)
        return {int(i): int(j) for i, j in zip(r, c)}

    def slot_of(self, i):
        return self.mapping()[i]

    def ident_at_slot(self, j):
        inv = {v: k for k, v in self.mapping().items()}
        return inv[j]


def ellipse_residual(Z, h):
    """Fit a centred conic z^T Q z = 1 through +-h (constraint) to points Z; return rms relative radial error."""
    Z = np.asarray(Z, float)
    if len(Z) < 2:
        return 1.0
    A = np.stack([Z[:, 0] ** 2, 2 * Z[:, 0] * Z[:, 1], Z[:, 1] ** 2], 1)
    c = np.array([h[0] ** 2, 2 * h[0] * h[1], h[1] ** 2])
    K = np.zeros((4, 4))
    K[:3, :3] = 2 * A.T @ A + 1e-9 * np.eye(3)
    K[:3, 3] = c
    K[3, :3] = c
    rhs = np.concatenate([2 * A.T @ np.ones(len(Z)), [1.0]])
    try:
        q = np.linalg.solve(K, rhs)[:3]
    except np.linalg.LinAlgError:
        return 1.0
    Q = np.array([[q[0], q[1]], [q[1], q[2]]])
    if np.linalg.eigvalsh(Q).min() <= 0:
        return 1.0
    v = np.einsum('ni,ij,nj->n', Z, Q, Z)
    # radial error: r_obs / r_fit - 1 = sqrt(v) - 1
    return float(np.sqrt(np.mean((np.sqrt(np.maximum(v, 0)) - 1.0) ** 2)))


def detect_swaps(frames, slots, log=print, shape=None, tol_frac=0.2, min_tol=5.0, rho_tol=0.22, debug=False):
    """Find pair swaps: two targets rotating 180 deg about the midpoint of their slots.
    A frame supports pair (a, b) when two blobs, both away from slots a and b, are antipodal about the
    pair midpoint and lie on the projected circle through the two slots.
    Returns a time-ordered list of dicts {a, b, t, t0, t1, n}."""
    S = np.array(slots)
    ns = len(S)
    if shape is None:
        shape = slot_shape(frames[0][1], slots)
    Minv = np.linalg.inv(shape)

    def mnorm(v):
        return float(np.sqrt(max(v @ Minv @ v, 0.0)))

    pairs = [(a, b) for a in range(ns) for b in range(a + 1, ns)]
    mids = {p: (S[p[0]] + S[p[1]]) / 2.0 for p in pairs}
    half = {p: np.linalg.norm(S[p[0]] - S[p[1]]) / 2.0 for p in pairs}
    mrad = {p: mnorm(S[p[0]] - mids[p]) for p in pairs}
    support = {p: [] for p in pairs}
    for fi, (t, front, _w) in enumerate(frames):
        bl = [np.array(b_[:2]) for b_ in target_blobs(front) if b_[2] >= 20]
        if len(bl) < 2:
            continue
        hyps = []
        for p in pairs:
            m, h = mids[p], half[p]
            # vacancy: a target still sitting on either slot rules the pair out
            if any(np.linalg.norm(x - S[p[0]]) < 4.0 or np.linalg.norm(x - S[p[1]]) < 4.0 for x in bl):
                continue
            away = max(6.0, 0.25 * h)
            tol = max(min_tol, tol_frac * h)
            u = (S[p[1]] - S[p[0]]) / (2.0 * h)
            cand = [i for i, x in enumerate(bl) if np.linalg.norm(x - S[p[0]]) > away and np.linalg.norm(x - S[p[1]]) > away
                    and abs((x - m) @ u) < 1.15 * h + 3.0 and np.linalg.norm(x - m) < 4.0 * h + 25.0]
            for ii in range(len(cand)):
                for jj in range(ii + 1, len(cand)):
                    i, j = cand[ii], cand[jj]
                    d = np.linalg.norm(bl[i] + bl[j] - 2 * m) / 2.0
                    if d < tol:
                        hyps.append((d / tol, p, i, j, d, bl[i] - m, bl[j] - m))
        for score, p, i, j, d, zi, zj in hyps:
            support[p].append((fi, t, d, zi, zj))
    events = []
    for p, sup in support.items():
        if not sup:
            continue
        grp = [sup[0]]
        for x in sup[1:]:
            if x[1] - grp[-1][1] > 1.0:
                events.append((p, grp)); grp = [x]
            else:
                grp.append(x)
        events.append((p, grp))
    cand = []
    for p, grp in events:
        h = (S[p[0]] - S[p[1]]) / 2.0
        Z = [g[3] for g in grp] + [g[4] for g in grp]
        res = ellipse_residual(Z, h)
        hn = np.linalg.norm(h)
        u = h / hn
        al = [abs(z @ u) / hn for z in Z]   # along-chord coordinate (1 at the slots, 0 half way)
        sweep = max(al) - min(al)
        dur = grp[-1][1] - grp[0][1]
        cand.append({'a': p[0], 'b': p[1], 't': float(np.mean([g[1] for g in grp])),
                     't0': grp[0][1], 't1': grp[-1][1], 'n': len(grp), 'err': float(np.mean([g[2] for g in grp])),
                     'res': res, 'sweep': float(sweep), 'dur': dur,
                     'score': len(grp) * max(0.0, 1.0 - res / 0.35)})
    dt_med = float(np.median(np.diff([f[0] for f in frames]))) if len(frames) > 2 else 0.05
    # a real swap lasts ~1.5 s: demand enough distinct frames to over-determine the conic (2 free params),
    # a plausible duration and a sweep from near a slot towards the middle
    exp_n = 1.3 / max(dt_med, 0.05)
    min_n = max(3, min(6, int(0.35 * exp_n)))
    for c in cand:
        c['ok'] = (c['n'] >= min_n and c['score'] > 0 and 0.4 <= c['dur'] + dt_med <= 2.6 and c['sweep'] >= 0.35)
    if debug:
        for c in sorted(cand, key=lambda c: c['t']):
            log(f"cand {c['a']}{c['b']} t={c['t0']:.2f}-{c['t1']:.2f} n={c['n']} err={c['err']:.1f} res={c['res']:.3f} "
                f"sweep={c['sweep']:.2f} score={c['score']:.1f} ok={c['ok']}")
    cand = [c for c in cand if c['ok']]
    cand.sort(key=lambda c: (-c['score'], c['err']))
    kept = []
    for c in cand:
        clash = False
        for k in kept:
            if {c['a'], c['b']} & {k['a'], k['b']} and not (c['t1'] < k['t0'] - 0.3 or c['t0'] > k['t1'] + 0.3):
                clash = True
                break
        if not clash:
            kept.append(c)
    kept.sort(key=lambda c: c['t'])
    return kept


def detect_placements(frames, slots, color, log=print):
    """Placement events: the goal cube rests on a slot and is released (wrist view no longer filled by it,
    or - without wrist frames - the cube is stationary)."""
    events = []
    cur = None
    prev_cube = None
    prev_t = frames[0][0]
    for fi, (t, front, wrist) in enumerate(frames):
        dt = max((t - prev_t) * 20.0, 1e-3) if fi else 1.0
        prev_t = t
        cb = cube_blob(front, color)
        held = None
        if wrist is not None:
            wn = int(cube_mask(wrist, color).sum())
            held = wn > 0.06 * wrist.shape[0] * wrist.shape[1]
        on_slot = None
        if cb is not None:
            ds = [np.hypot(s[0] - cb[0], s[1] - (cb[1] - 2.0)) for s in slots]
            j = int(np.argmin(ds))
            if ds[j] < 11.0:
                on_slot = j
        moving = True
        if cb is not None and prev_cube is not None:
            moving = np.hypot(cb[0] - prev_cube[0], cb[1] - prev_cube[1]) > 1.5 * max(1.0, dt / 3.0)
        prev_cube = cb
        released = (held is False) or (held is None and not moving)
        if on_slot is not None and released:
            if cur is not None and cur['slot'] == on_slot:
                cur['t1'] = t; cur['n'] += 1
            else:
                if cur is not None and cur['n'] >= 3:
                    events.append(cur)
                cur = {'slot': on_slot, 't0': t, 't1': t, 'n': 1}
        else:
            if cur is not None and (cb is None and held is None):
                pass
            elif cur is not None:
                if cur['n'] >= 3:
                    events.append(cur)
                cur = None
    if cur is not None and cur['n'] >= 3:
        events.append(cur)
    merged = []
    for e in events:
        if merged and merged[-1]['slot'] == e['slot'] and e['t0'] - merged[-1]['t1'] < 1.5:
            merged[-1]['t1'] = e['t1']; merged[-1]['n'] += e['n']
        else:
            merged.append(e)
    return merged


def analyze_video(frames, color, k, log=print, debug=False):
    """frames: list of (timestamp_s, front_rgb, wrist_rgb_or_None) in order (conditioning video, optionally
    followed by the first execution frame). Returns dict with 'answer_slot' / 'answer_xy' (front pixel)."""
    slots = find_slots(frames, log)
    log(f'vision: {len(slots)} slots: ' + ' '.join(f'{j}:({s[0]:.0f},{s[1]:.0f})' for j, s in enumerate(slots)))
    swaps = detect_swaps(frames, slots, log=log, debug=debug)
    log('vision: swaps: ' + ' '.join(f"{c['a']}<->{c['b']}@{c['t']:.1f}s(n{c['n']},r{c['res']:.2f})" for c in swaps))
    places = detect_placements(frames, slots, color, log)
    # timeline: perm[slot] = identity (identity = slot index at video start)
    perm = list(range(len(slots)))
    timeline = [(c['t'], 'swap', c) for c in swaps] + [((e['t0'] + e['t1']) / 2.0, 'place', e) for e in places]
    timeline.sort(key=lambda x: x[0])
    for t, kind, e in timeline:
        if kind == 'swap':
            a, b = e['a'], e['b']
            perm[a], perm[b] = perm[b], perm[a]
        else:
            e['ident'] = perm[e['slot']]
    final_slot_of = {ident: j for j, ident in enumerate(perm)}
    log('vision: placements: ' + ', '.join(
        f"#{n + 1} slot{e['slot']} id{e['ident']} t={e['t0']:.1f}-{e['t1']:.1f}" for n, e in enumerate(places)))
    res = {'slots': [s.tolist() for s in slots], 'swaps': swaps, 'events': places, 'final_perm': perm}
    if not places:
        res['answer_slot'] = None
        return res
    kk = k if k is not None else len(places)
    if kk > len(places):
        log(f'vision: only {len(places)} placements found, k={kk}; using last')
        kk = len(places)
    e = places[kk - 1]
    res['answer_ident'] = e['ident']
    res['answer_slot'] = final_slot_of[e['ident']]
    res['answer_xy'] = slots[res['answer_slot']].tolist()
    res['k_used'] = kk
    log(f"vision: k={kk} -> identity {e['ident']} (placed on slot {e['slot']}) now at slot {res['answer_slot']} "
        f"({res['answer_xy'][0]:.0f},{res['answer_xy'][1]:.0f})")
    return res
