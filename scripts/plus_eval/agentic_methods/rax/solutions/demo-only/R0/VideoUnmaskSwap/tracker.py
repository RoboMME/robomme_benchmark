"""Container tracking in the front camera of the conditioning video (pure numpy, 2D image space).

Input: list of (t, rgb uint8 HxWx3) front frames in time order; frame 0 must be the empty table.
Output: dict with cube colors, final container positions (pixel) with color labels.
"""
import numpy as np

# ---- tunables (pixel units at 256x256; tested on 3 demos) ----
FG_DIFF = 40          # min max-channel difference from background to be foreground
SAT_MAX = 45          # container pixels: max-min channel below this
BRIGHT_MIN = 110      # container pixels: min channel... (mean) above this
CUBE_MIN_AREA = 12    # min pixel area of a cube blob
COVER_JUMP = 2.0      # container-pixel count ratio indicating cover event


def _components(mask, min_area=1):
    """4-connected components; returns list of (ys, xs) arrays."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    comps = []
    cur = 0
    ys0, xs0 = np.nonzero(mask)
    for y0, x0 in zip(ys0, xs0):
        if lab[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]
        lab[y0, x0] = cur
        py, px = [], []
        while stack:
            y, x = stack.pop()
            py.append(y); px.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = cur
                    stack.append((yy, xx))
        if len(py) >= min_area:
            comps.append((np.array(py), np.array(px)))
    return comps


def fg_mask(img, bg):
    d = np.abs(img.astype(int) - bg.astype(int)).max(2)
    return d > FG_DIFF


def container_mask(img, bg):
    im = img.astype(int)
    sat = im.max(2) - im.min(2)
    bright = im.mean(2)
    return fg_mask(img, bg) & (sat < SAT_MAX) & (bright > BRIGHT_MIN)


def classify_color(rgb):
    r, g, b = [float(v) for v in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    sat = mx - mn
    if mx < 70:
        return 'black'
    if sat < 40:
        return 'white' if mx > 140 else ('gray' if mx > 90 else 'black')
    # hue
    import colorsys
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    h *= 360
    if h < 15 or h >= 330:
        return 'red'
    if h < 45:
        return 'orange'
    if h < 70:
        return 'yellow'
    if h < 170:
        return 'green'
    if h < 255:
        return 'blue'
    return 'purple'


def find_cover(frames, bg):
    """Index of first frame where containers have appeared."""
    counts = [int(container_mask(im, bg).sum()) for _, im in frames]
    base = max(1, np.median(counts[1:6]) if len(counts) > 6 else 1)
    for i in range(1, len(counts)):
        if counts[i] > COVER_JUMP * base + 150:
            return i, counts
    return None, counts


def detect_cubes(img, bg):
    m = fg_mask(img, bg)
    out = []
    for ys, xs in _components(m, CUBE_MIN_AREA):
        rgb = np.median(img[ys, xs], axis=0)
        # use brightest-ish saturated pixels for color
        out.append(dict(c=np.array([xs.mean(), ys.mean()]), area=len(ys), rgb=rgb.tolist(),
                        color=classify_color(rgb), bottom=float(ys.max())))
    return out


def hungarian(cost):
    """Small Hungarian via scipy if present else brute-force greedy."""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        cost = np.array(cost, float).copy()
        pairs = []
        n, m = cost.shape
        for _ in range(min(n, m)):
            i, j = np.unravel_index(np.argmin(cost), cost.shape)
            pairs.append((int(i), int(j)))
            cost[i, :] = np.inf
            cost[:, j] = np.inf
        return pairs


class Tracker:
    def __init__(self, centers, radius):
        self.c = np.array(centers, float)      # (N,2) x,y
        self.v = np.zeros_like(self.c)
        self.radius = radius
        self.lost = np.zeros(len(self.c), int)

    def update(self, mask, iters=6):
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            self.lost += 1
            return
        P = np.stack([xs, ys], 1).astype(float)
        pred = self.c + 0.5 * self.v
        c = pred.copy()
        R = self.radius
        for _ in range(iters):
            d = np.linalg.norm(P[:, None, :] - c[None], axis=2)  # (M,N)
            a = d.argmin(1)
            dm = d[np.arange(len(P)), a]
            newc = c.copy()
            got = np.zeros(len(c), bool)
            for k in range(len(c)):
                sel = (a == k) & (dm < R)
                if sel.sum() >= 8:
                    newc[k] = P[sel].mean(0)
                    got[k] = True
            c = newc
        for k in range(len(c)):
            if got[k]:
                self.v[k] = 0.7 * self.v[k] + 0.3 * (c[k] - self.c[k])
                self.c[k] = c[k]
                self.lost[k] = 0
            else:
                self.c[k] = pred[k]
                self.v[k] *= 0.5
                self.lost[k] += 1


def track(frames, log=None):
    """frames: list of (t, img). Returns dict."""
    bg = frames[0][1]
    ci, counts = find_cover(frames, bg)
    if ci is None:
        raise RuntimeError('cover event not found')
    # cubes: the frame 2 samples before cover (stable)
    cube_idx = max(1, ci - 2)
    cubes = detect_cubes(frames[cube_idx][1], bg)
    # containers at first stable cover frame
    m0 = container_mask(frames[ci][1], bg)
    comps = _components(m0, 30)
    cont = [np.array([xs.mean(), ys.mean()]) for ys, xs in comps]
    areas = [len(ys) for ys, xs in comps]
    med_area = float(np.median(areas)) if areas else 300.0
    size = np.sqrt(med_area)
    # number of containers = number of cubes (each cube hidden)
    N = len(cubes)
    # init track centres: cube centre + median offset to the nearest container blob
    cc = np.array([q['c'] for q in cubes])
    if cont:
        C = np.array(cont)
        d = np.linalg.norm(cc[:, None] - C[None], axis=2)
        offs = C[d.argmin(1)] - cc
        off = np.median(offs, 0)
    else:
        off = np.zeros(2)
    init = cc + off
    tr = Tracker(init, radius=0.9 * size)
    # refine on cover frame a few times
    for _ in range(2):
        tr.update(m0)
    hist = [tr.c.copy()]
    for t, im in frames[ci + 1:]:
        tr.update(container_mask(im, bg))
        hist.append(tr.c.copy())
    return dict(tracker=tr, cover_index=ci, cube_index=cube_idx, cubes=cubes, offset=off.tolist(), size=size,
                final=tr.c.copy(), colors=[q['color'] for q in cubes], hist=np.array(hist),
                lost=tr.lost.copy(), n=N)


def static_mask(bg, exec0, tracks, size):
    """Pixels to ignore in execution frames: static low-sat objects and the robot at its exec pose."""
    def lowsat(im):
        im = im.astype(int)
        return ((im.max(2) - im.min(2)) < SAT_MAX) & (im.mean(2) > BRIGHT_MIN)
    ls_bg, ls_ex = lowsat(bg), lowsat(exec0)
    same = np.abs(bg.astype(int) - exec0.astype(int)).max(2) < FG_DIFF
    static = ls_bg & ls_ex & same
    # robot at its execution pose: low-sat components of exec0 touching the top border
    H, W = ls_ex.shape
    for ys, xs in _components(ls_ex, 1):
        if ys.min() <= 1:
            static[ys, xs] = True
    # do not mask pixels right at the tracked container centres
    yy, xx = np.mgrid[0:H, 0:W]
    for x, y in tracks:
        static[(xx - x) ** 2 + (yy - y) ** 2 < (0.35 * size) ** 2] = False
    return static


def exec_mask(img, static):
    im = img.astype(int)
    return ((im.max(2) - im.min(2)) < SAT_MAX) & (im.mean(2) > BRIGHT_MIN) & ~static
