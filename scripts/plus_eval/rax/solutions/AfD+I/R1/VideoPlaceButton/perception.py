"""Offline-testable perception for VideoPlaceButton.

All functions take front-camera RGB frames (H x W x 3 uint8/int arrays, 256x256) from the
conditioning video and return pixel-space results. The front camera is static, so pixel
coordinates of the video and of the live execution frame coincide.

Pipeline (analyse_video):
  1. target mask per frame (pink/purple bullseyes: B > G and R > G, not wood/grey/cube)
  2. homes = resting spots of targets (pixels that are target in most frames)
  3. per-frame target detections (blob splitting by expected disc area)
  4. multi-target tracking with constant-velocity prediction + Hungarian matching,
     snapping to homes whenever the scene is at rest
  5. cube (instructed colour) track, rest periods
  6. button press time = longest strong occlusion/change of the button region
  7. first cube placement after the press -> home -> target identity -> its final home
"""
import math
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------- tunables
TARGET_BG_MIN = 8        # B - G > this (pink 18..25, purple ~100; wood is negative)
TARGET_RG_MIN = 25       # R - G > this
TARGET_R_MIN = 100
TARGET_SAT_MIN = 0.12
TARGET_SAT_MAX = 0.75
WOOD_HUE_LO = 6.0         # deg; wood 18-24
WOOD_HUE_HI = 62.0
HOME_FREQ = 0.45         # fraction of frames a pixel must be target to be a home pixel
MIN_BLOB = 15            # px, smallest target fragment kept
REST_SPEED = 0.6         # px/frame below which a track counts as resting
SNAP_DIST = 7.0          # px, a detection within this of a home is "at home"
GATE_BASE = 14.0         # px, matching gate (plus speed term)
PLACE_MIN_S = 0.8        # s, a placement rest lasts at least this (demos: 5-20 s; carry pauses 0.6-0.8 s)
CUBE_ON_HOME_DX = 0.0    # px, cube centroid minus home centre when resting on it (demos -1..4)
CUBE_ON_HOME_DY = -3.5   # px (demos -2..-4.3)
PLACE_TOL_PX = 7.0       # px (true placements <= 4.5, carry pause 11)


def _hsv(im):
    im = im.astype(np.float32)
    mx = im.max(-1)
    mn = im.min(-1)
    c = mx - mn
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    h = np.zeros_like(mx)
    nz = c > 1e-3
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = (60.0 * (g[rm] - b[rm]) / c[rm]) % 360.0
    h[gm] = 60.0 * (b[gm] - r[gm]) / c[gm] + 120.0
    h[bm] = 60.0 * (r[bm] - g[bm]) / c[bm] + 240.0
    sat = np.where(mx > 0, c / np.maximum(mx, 1e-3), 0)
    return h, sat, mx


def any_cube_mask(im):
    return cube_mask(im, "red") | cube_mask(im, "green") | cube_mask(im, "blue")


def table_region(frames):
    """Pixels on the wooden table (holes filled), from the median frame."""
    sub = frames[:: max(1, len(frames) // 40)]
    med = np.median(np.stack(sub).astype(np.float32), 0).astype(np.uint8)
    w = wood_mask(med)
    w = ndimage.binary_closing(w, iterations=6)
    w = ndimage.binary_fill_holes(w)
    w = ndimage.binary_erosion(w, iterations=2)
    return w


def target_mask(im, region=None):
    """Bullseye targets: moderately saturated colour whose hue is outside the wood band.
    Demos: pink hue ~340 / purple ~298, sat 0.15-0.6; wood hue 18-24, sat 0.5-0.75;
    cubes sat ~1."""
    h, sat, v = _hsv(im)
    m = (sat > TARGET_SAT_MIN) & (sat < TARGET_SAT_MAX) & (v > 90) & \
        ~((h > WOOD_HUE_LO) & (h < WOOD_HUE_HI))
    m = ndimage.binary_closing(m, iterations=1)
    m = ndimage.binary_fill_holes(m)
    m &= ~ndimage.binary_dilation(any_cube_mask(im), iterations=1)
    m = ndimage.binary_opening(m, iterations=1)
    if region is not None:
        m &= region
    return m


def cube_mask(im, color):
    im = im.astype(np.int16)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    if color == "red":
        m = (r > 80) & (g < 60) & (b < 60) & (r > 2 * g + 40)
    elif color == "blue":
        m = (b > 80) & (r < 60) & (g < 60) & (b > 2 * g + 40)
    elif color == "green":
        m = (g > 80) & (r < 60) & (b < 60) & (g > 2 * r + 40)
    else:
        raise ValueError(color)
    return m


def wood_mask(im):
    im = im.astype(np.int16)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (r - b > 50) & (g - b > 10) & (r > g) & (g > b)


# ---------------------------------------------------------------- homes
def find_homes(masks):
    """Home centres (x, y) and areas from the per-pixel target frequency."""
    freq = np.mean(np.stack(masks).astype(np.float32), axis=0)
    hm = freq > HOME_FREQ
    hm = ndimage.binary_opening(hm, iterations=1)
    lab, n = ndimage.label(hm)
    homes = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if len(xs) < 25:
            continue
        # split touching homes using distance-transform peaks
        sub = lab == k
        dist = ndimage.distance_transform_edt(sub)
        peaks = (dist == ndimage.maximum_filter(dist, size=9)) & (dist >= 3.0) & sub
        py, px = np.nonzero(peaks)
        pts = []
        for x, y in sorted(zip(px, py), key=lambda p: -dist[p[1], p[0]]):
            if all(math.hypot(x - a, y - b) > 9 for a, b in pts):
                pts.append((x, y))
        if len(pts) <= 1:
            homes.append([xs.mean(), ys.mean(), float(len(xs))])
        else:
            c = np.array(pts, dtype=float)
            P = np.stack([xs, ys], 1).astype(float)
            for _ in range(10):
                d = ((P[:, None, :] - c[None]) ** 2).sum(-1)
                a = d.argmin(1)
                c = np.array([P[a == j].mean(0) if np.any(a == j) else c[j] for j in range(len(c))])
            for j in range(len(c)):
                homes.append([c[j, 0], c[j, 1], float(np.sum(a == j))])
    return np.array(homes, dtype=float).reshape(-1, 3), freq


def detect_targets(mask, homes):
    """Target centre detections in one frame; big blobs are split into k discs."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0)
    dets, areas = [], []
    objs = ndimage.find_objects(lab)
    for k in range(1, n + 1):
        sl = objs[k - 1]
        sub = lab[sl] == k
        ys, xs = np.nonzero(sub)
        if len(xs) < MIN_BLOB:
            continue
        xs = xs + sl[1].start
        ys = ys + sl[0].start
        cx, cy = xs.mean(), ys.mean()
        # expected single-target area near here
        if len(homes):
            d = np.hypot(homes[:, 0] - cx, homes[:, 1] - cy)
            w = np.exp(-d / 25.0) + 1e-6
            exp_area = float(np.sum(w * homes[:, 2]) / np.sum(w))
        else:
            exp_area = 250.0
        kk = int(round(len(xs) / max(exp_area, 50.0)))
        kk = max(1, min(kk, 4))
        if kk == 1:
            dets.append((cx, cy)); areas.append(len(xs))
            continue
        P = np.stack([xs, ys], 1).astype(float)
        # farthest-point init
        c = [P[np.argmax(((P - P.mean(0)) ** 2).sum(1))]]
        for _ in range(kk - 1):
            dd = np.min(((P[:, None, :] - np.array(c)[None]) ** 2).sum(-1), 1)
            c.append(P[np.argmax(dd)])
        c = np.array(c)
        for _ in range(12):
            d2 = ((P[:, None, :] - c[None]) ** 2).sum(-1)
            a = d2.argmin(1)
            c = np.array([P[a == j].mean(0) if np.any(a == j) else c[j] for j in range(kk)])
        for j in range(kk):
            dets.append((c[j, 0], c[j, 1])); areas.append(int(np.sum(a == j)))
    return np.array(dets, dtype=float).reshape(-1, 2), np.array(areas, dtype=float)


# ---------------------------------------------------------------- tracking
class TargetTracker:
    """Tracks N identical targets; identity i starts at home i."""

    def __init__(self, homes):
        self.homes = homes[:, :2].copy()
        n = len(homes)
        self.pos = self.homes.copy()
        self.vel = np.zeros((n, 2))
        self.miss = np.zeros(n, dtype=int)
        self.home_of = np.arange(n)          # current home index of each identity (at rest)
        self.history = []                     # (frame_index, home_of copy) at rest moments
        self.events = []

    def step(self, t_idx, dets, dt):
        n = len(self.pos)
        pred = self.pos + self.vel * dt
        assigned = np.full(n, -1)
        if len(dets):
            cost = np.hypot(pred[:, None, 0] - dets[None, :, 0], pred[:, None, 1] - dets[None, :, 1])
            speed = np.hypot(self.vel[:, 0], self.vel[:, 1]) * dt
            gate = GATE_BASE + 1.2 * speed + 3.0 * self.miss
            big = 1e6
            c2 = np.where(cost <= gate[:, None], cost, big)
            r, c = linear_sum_assignment(c2)
            for i, j in zip(r, c):
                if c2[i, j] < big:
                    assigned[i] = j
        for i in range(n):
            if assigned[i] >= 0:
                newp = dets[assigned[i]]
                v = (newp - self.pos[i]) / max(dt, 1e-6)
                self.vel[i] = 0.5 * self.vel[i] + 0.5 * v if self.miss[i] == 0 else v
                self.pos[i] = newp
                self.miss[i] = 0
            else:
                self.pos[i] = pred[i]
                self.vel[i] *= 0.85
                self.miss[i] += 1
        # rest check: every detection is at a home and every track slow -> snap
        at_home = True
        if len(dets):
            dh = np.min(np.hypot(dets[:, None, 0] - self.homes[None, :, 0],
                                 dets[:, None, 1] - self.homes[None, :, 1]), 1)
            at_home = bool(np.all(dh < SNAP_DIST))
        speeds = np.hypot(self.vel[:, 0], self.vel[:, 1])
        if at_home and np.all(speeds < REST_SPEED * 3):
            cost = np.hypot(self.pos[:, None, 0] - self.homes[None, :, 0],
                            self.pos[:, None, 1] - self.homes[None, :, 1])
            r, c = linear_sum_assignment(cost)
            if np.max(cost[r, c]) < 16:
                new_home = np.empty(n, dtype=int)
                new_home[r] = c
                if not np.array_equal(new_home, self.home_of):
                    self.events.append((t_idx, self.home_of.copy(), new_home.copy()))
                self.home_of = new_home
                self.pos = self.homes[new_home].copy()
                self.vel[:] = 0
                self.history.append((t_idx, new_home.copy()))
        return at_home


# ---------------------------------------------------------------- cube & button
def blob_centroid(mask, min_px=20):
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < min_px:
        return None
    ys, xs = np.nonzero(lab == k)
    return (float(xs.mean()), float(ys.mean()), int(len(xs)), float(ys.max()))


def find_button(im):
    """White square button with grey cap: bright low-saturation blob below the robot area."""
    im = im.astype(np.int16)
    mx, mn = im.max(-1), im.min(-1)
    white = (mn > 150) & (mx - mn < 30)
    white[:40, :] = False  # robot base / far edge
    white = ndimage.binary_closing(white, iterations=2)
    white = ndimage.binary_fill_holes(white)
    lab, n = ndimage.label(white)
    best = None
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if not (40 <= len(xs) <= 900):
            continue
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        fill = len(xs) / float(h * w)
        if fill < 0.6 or not (0.5 < w / h < 2.2):
            continue
        score = len(xs) * fill
        if best is None or score > best[0]:
            best = (score, xs.min(), ys.min(), xs.max(), ys.max())
    if best is None:
        return None
    return tuple(int(v) for v in best[1:])


def rest_periods(track, times, tol=2.5, min_dur=0.6):
    """Intervals where the cube centroid stays within tol px. track: list of (x,y) or None."""
    periods = []
    i, n = 0, len(track)
    while i < n:
        if track[i] is None:
            i += 1
            continue
        j = i
        ref = np.array(track[i][:2])
        pts = [ref]
        while j + 1 < n and track[j + 1] is not None and \
                np.hypot(*(np.array(track[j + 1][:2]) - np.mean(pts, 0))) < tol:
            j += 1
            pts.append(np.array(track[j][:2]))
        if times[j] - times[i] >= min_dur:
            c = np.median(np.array(pts), 0)
            periods.append((i, j, float(c[0]), float(c[1])))
        i = j + 1
    return periods


# ---------------------------------------------------------------- full analysis
def analyse_video(frames, times, color, log=print):
    """frames: list of front RGB arrays (video frames followed by the live frame).
    times: matching timestamps (s). Returns dict with the destination pixel and diagnostics."""
    n = len(frames)
    region = table_region(frames)
    log(f"table region px={int(region.sum())} rows {np.nonzero(region.any(1))[0][[0, -1]].tolist()}")
    masks = [target_mask(f, region) for f in frames]
    homes, freq = find_homes(masks)
    log(f"homes={np.round(homes, 1).tolist()}")
    res = {"homes": homes.tolist(), "ok": False}
    if len(homes) < 2:
        res["why"] = "fewer than 2 homes"
        return res

    # --- targets
    tr = TargetTracker(homes)
    rest_flags = []
    for k in range(n):
        dets, _ = detect_targets(masks[k], homes)
        dt_frames = 1.0 if k == 0 else max((times[k] - times[k - 1]) * 20.0, 1e-3)
        rest_flags.append(tr.step(k, dets, dt_frames))
    log(f"target events={[(e[0], e[1].tolist(), e[2].tolist()) for e in tr.events]}")

    # --- cube track (instructed colour)
    ctrack = []
    for f in frames:
        c = blob_centroid(cube_mask(f, color))
        ctrack.append(c)
    # --- button press
    btn = find_button(frames[0]) or find_button(frames[-1])
    res["button_box"] = btn
    press = None
    if btn is not None:
        x0, y0, x1, y1 = btn
        ref = np.median(np.stack([f[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
                                  for f in frames[:: max(1, n // 60)]]), 0)
        diffs = []
        for f in frames:
            patch = f[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
            d = np.abs(patch - ref).max(-1) > 30
            diffs.append(float(d.mean()))
        diffs = np.array(diffs)
        res["button_diff_max"] = float(diffs.max())
        # intervals of strong change
        strong = diffs > 0.15
        ivals = []
        k = 0
        while k < n:
            if strong[k]:
                j = k
                while j + 1 < n and strong[j + 1]:
                    j += 1
                ivals.append((k, j, float(diffs[k:j + 1].mean()), times[j] - times[k]))
                k = j + 1
            else:
                k += 1
        log(f"button box={btn} strong intervals={[(a, b, round(m, 2), round(d, 2)) for a, b, m, d in ivals]}")
        if ivals:
            # the press: strongest sustained occlusion; earliest among those near the max score
            # the press: the interval with the largest integrated change (time-weighted)
            scores = [float(np.sum(diffs[a:b + 1] * np.gradient(np.array(times[a:b + 2] if b + 2 <= n else times[a:b + 1] + [times[b] + 0.05]))[:b - a + 1]))
                      for a, b, m, d in ivals]
            press = ivals[int(np.argmax(scores))]
    if press is None:
        res["why"] = "no button press found"
        return res
    res["press"] = (press[0], press[1])

    # --- placements (cube rest periods)
    per = rest_periods(ctrack, times)
    log(f"cube rest periods={[(a, b, round(x), round(y)) for a, b, x, y in per]}")
    # merge consecutive rest periods at the same spot (gripper occlusion splits them)
    merged = []
    for a, b, x, y in per:
        if merged and np.hypot(x - merged[-1][2], y - merged[-1][3]) < 5:
            merged[-1] = (merged[-1][0], b, merged[-1][2], merged[-1][3])
        else:
            merged.append((a, b, x, y))
    res["cube_rests"] = merged
    place = None
    for a, b, x, y in merged:
        if a <= press[0]:
            continue
        if times[b] - times[a] < PLACE_MIN_S:
            continue
        dh = np.hypot(homes[:, 0] + CUBE_ON_HOME_DX - x, homes[:, 1] + CUBE_ON_HOME_DY - y)
        if dh.min() < PLACE_TOL_PX:
            place = (a, b, x, y, int(dh.argmin()))
            break
    if place is None:
        res["why"] = "no placement after press"
        return res
    a, b, x, y, h = place
    res["placement"] = {"frame": a, "end": b, "cube_px": [x, y], "home": h}
    # event-based swap inference (primary)
    events = infer_swaps(frames, masks, homes, times, log=log)
    final_home = follow_identity(events, h, a)
    res["events"] = [(int(e0), int(e1), [(int(p), int(q)) for p, q in pr]) for e0, e1, pr in events]
    # frame-tracker result (cross-check only)
    hist = [(t, ho) for t, ho in tr.history if t <= b]
    if hist:
        ident = int(np.where(hist[-1][1] == h)[0][0])
        res["tracker_final_home"] = int(tr.home_of[ident])
    res["identity"] = int(h)
    res["final_home"] = int(final_home)
    res["final_px"] = homes[final_home, :2].tolist()
    res["ok"] = True
    res["cube_final"] = ctrack[-1][:2] if ctrack[-1] else None
    log(f"press={press[:2]} placement={res['placement']} tracker={res.get('tracker_final_home')} final_home={final_home} "
        f"px={np.round(homes[final_home, :2], 1).tolist()}")
    return res


# ---------------------------------------------------------------- event-based swap inference
VAC_TARGET_MAX = 0.20    # home disk target fraction below this ...
VAC_WOOD_MIN = 0.45      # ... and wood fraction above this => home vacated
MOVE_MIN_DIST = 9.0      # px, a detection this far from every home is a moving target
REST_GAP = 1             # consecutive rest frames needed to close an event


def home_disks(homes, shape=(256, 256)):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    disks = []
    for x, y, a in homes:
        rad = max(3.0, 0.55 * math.sqrt(a / math.pi))
        disks.append((xx - x) ** 2 + (yy - y) ** 2 <= rad * rad)
    return disks


def _matchings(items):
    """All perfect matchings (as lists of pairs) of an even-sized list."""
    if not items:
        yield []
        return
    a = items[0]
    for i in range(1, len(items)):
        rest = items[1:i] + items[i + 1:]
        for m in _matchings(rest):
            yield [(a, items[i])] + m


def infer_swaps(frames, masks, homes, times, log=print):
    """Return list of events: (start_idx, end_idx, [(h1, h2), ...]) of 180-degree pair swaps."""
    n = len(frames)
    H = homes[:, :2]
    disks = home_disks(homes)
    vac = np.zeros((n, len(H)), dtype=bool)
    moving = []      # per frame: array of moving detections
    for k in range(n):
        m = masks[k]
        wood = wood_mask(frames[k])
        for h, dk in enumerate(disks):
            tot = dk.sum()
            vac[k, h] = (m[dk].sum() / tot < VAC_TARGET_MAX) and (wood[dk].sum() / tot > VAC_WOOD_MIN)
        dets, _ = detect_targets(m, homes)
        if len(dets):
            dh = np.min(np.hypot(dets[:, None, 0] - H[None, :, 0], dets[:, None, 1] - H[None, :, 1]), 1)
            moving.append(dets[dh > MOVE_MIN_DIST])
        else:
            moving.append(np.zeros((0, 2)))
    active = vac.any(1) | np.array([len(mv) > 0 for mv in moving])
    # events = runs of active frames (bridging single-frame gaps)
    events = []
    k = 0
    while k < n:
        if not active[k]:
            k += 1
            continue
        j = k
        while j + 1 < n and (active[j + 1] or (j + 2 < n and active[j + 2])):
            j += 1
        events.append((k, j))
        k = j + 1
    out = []
    for a, b in events:
        V = [h for h in range(len(H)) if vac[a:b + 1, h].any()]
        mv = [(t, d) for t in range(a, b + 1) for d in moving[t]]
        if len(V) < 2:
            if mv:
                log(f"event {a}-{b}: vacated={V} moving={len(mv)} -> ignored")
            continue
        # candidate pairs scored by moving detections lying on their orbit
        def pair_score(p, q):
            c = (H[p] + H[q]) / 2
            r = np.hypot(*(H[p] - H[q])) / 2
            s = 0.0
            for t, d in mv:
                rho = np.hypot(*(d - c)) / r
                if 0.55 <= rho <= 1.35:
                    s += 1.0
                    partner = 2 * c - d
                    others = moving[t]
                    if len(others) and np.min(np.hypot(others[:, 0] - partner[0], others[:, 1] - partner[1])) < 6 + 0.3 * r:
                        s += 1.0
            # co-vacancy: both homes empty at the same time
            s += 2.0 * float(np.sum(vac[a:b + 1, p] & vac[a:b + 1, q]))
            return s
        cand = V[:]
        if len(cand) % 2 == 1:
            # one vacated home missed: add the best partner from the others
            best = None
            for h in range(len(H)):
                if h in cand:
                    continue
                sc = max(pair_score(h, v) for v in cand)
                if best is None or sc > best[0]:
                    best = (sc, h)
            if best is not None:
                cand.append(best[1])
        if len(cand) > 8:
            log(f"event {a}-{b}: too many vacated {cand}")
            cnt = {h: int(vac[a:b + 1, h].sum()) for h in cand}
            cand = sorted(cand, key=lambda h: -cnt[h])[:8]
        if len(cand) % 2 == 1:
            cnt = {h: int(vac[a:b + 1, h].sum()) for h in cand}
            cand.remove(min(cand, key=lambda h: cnt[h]))
        if len(cand) < 2:
            log(f"event {a}-{b}: vacated={V} unpaired -> ignored")
            continue
        best = None
        for mt in _matchings(sorted(cand)):
            sc = sum(pair_score(p, q) for p, q in mt)
            if best is None or sc > best[0]:
                best = (sc, mt)
        log(f"event {a}-{b} t={times[a]:.1f}-{times[b]:.1f}s vacated={V} moving={len(mv)} pairs={best[1]} score={best[0]:.0f}")
        out.append((a, b, best[1]))
    return out


def follow_identity(events, home, after_idx):
    """Follow the target that is at `home` at frame after_idx through later swap events."""
    h = home
    for a, b, pairs in events:
        # the target cannot swap while the cube rests on it: only events starting after the
        # placement began can move it
        if a <= after_idx:
            continue
        for p, q in pairs:
            if h == p:
                h = q
                break
            if h == q:
                h = p
                break
    return h
