"""Offline-capable shell-game tracker for VideoRepick.
Input: list of front RGB frames (HxWx3 uint8) of conditioning video, in order.
Output: index of picked object and its pixel position in the final frame."""
import numpy as np
from scipy import ndimage

def rgb2hsv(a):
    a = a.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1); d = mx - mn + 1e-6
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    s = (mx - mn) / (mx + 1e-6)
    return h, s, mx

def hue_dist(h1, h2):
    d = np.abs(h1 - h2) % 360
    return np.minimum(d, 360 - d)

class Seg:
    """Local-contrast colour segmentation: object pixels are saturated and differ in hue
    from a 25px median-filtered local background (removes table, light patches, static gray)."""
    def __init__(self, frames=None):
        self.obj_hue = None
        self.excl = None

    def build_exclusion(self, frames):
        """Static gray/white structures (button, parked robot, pillars) that light can tint."""
        acc = None
        for f in frames:
            h, s, v = rgb2hsv(f)
            g = (s < 0.2) & (v > 0.25)
            acc = g.astype(np.float32) if acc is None else acc + g
        self.excl = ndimage.binary_dilation(acc / len(frames) > 0.3, iterations=2)

    def raw(self, f):
        h, s, v = rgb2hsv(f)
        small = f[::4, ::4]
        loc = np.stack([ndimage.median_filter(small[..., c], size=7) for c in range(3)], -1)
        loc = np.repeat(np.repeat(loc, 4, 0), 4, 1)[: f.shape[0], : f.shape[1]]
        lh, ls, lv = rgb2hsv(loc)
        gh = np.median(h[s > 0.2]) if (s > 0.2).any() else 20.0
        m = (s > 0.3) & (hue_dist(h, lh) > 30) & (v > 0.12) & (hue_dist(h, gh) > 25)
        m &= ~((v > 0.72) & (s < 0.5))   # bright, weakly tinted robot links
        m[:22] = False                    # robot base region at the top of the front view
        return m, h

    def calibrate(self, frames):
        hs = []
        for f in frames:
            m, h = self.raw(f)
            hs.append(h[m])
        hs = np.concatenate(hs) if hs else np.zeros(0)
        if len(hs) < 20:
            self.obj_hue = None; return
        hist, edges = np.histogram(hs, bins=36, range=(0, 360))
        sm = hist + np.roll(hist, 1) + np.roll(hist, -1)
        k = sm.argmax(); self.obj_hue = (edges[k] + edges[k + 1]) / 2

    def mask(self, f):
        m, h = self.raw(f)
        if self.obj_hue is not None:
            m &= hue_dist(h, self.obj_hue) < 35
        if self.excl is not None:
            m &= ~self.excl
        m = ndimage.binary_opening(m, iterations=1)
        return m

def count_objects(m, min_area=10):
    lab, n = ndimage.label(m)
    if n == 0: return [], []
    areas = ndimage.sum(m, lab, range(1, n + 1))
    cents = ndimage.center_of_mass(m, lab, range(1, n + 1))
    keep = [(c, a) for c, a in zip(cents, areas) if a >= min_area]
    return [np.array(c) for c, a in keep], [a for c, a in keep]

def init_tracks(seg, frames):
    """Use early static frames; split merged blobs by area."""
    best = None
    for f in frames:
        m = seg.mask(f)
        cents, areas = count_objects(m)
        if not cents: continue
        med = np.median(areas)
        n = int(sum(min(3, max(1, round(a / med))) for a in areas))
        if n > 16: continue
        if best is None or n > best[0] or (n == best[0] and len(cents) > len(best[1])):
            best = (n, cents, areas, m)
    n, cents, areas, m = best
    ys, xs = np.nonzero(m); P = np.stack([ys, xs], 1).astype(np.float32)
    C = np.array(cents, np.float32)
    # add extra seeds for merged blobs
    med = np.median(areas)
    extra = []
    for c, a in zip(cents, areas):
        for j in range(int(min(3, max(1, round(a / med)))) - 1):
            extra.append(c + np.random.RandomState(j).randn(2) * 2)
    if extra: C = np.concatenate([C, np.array(extra, np.float32)])
    C = kmeans(P, C, iters=10)
    return C, med

def kmeans(P, C, iters=5, radius=14.0):
    C = C.copy()
    for _ in range(iters):
        d = ((P[:, None, :] - C[None]) ** 2).sum(-1)
        a = d.argmin(1); dm = d.min(1)
        ok = dm < radius ** 2
        for k in range(len(C)):
            sel = ok & (a == k)
            if sel.sum() >= 3:
                C[k] = P[sel].mean(0)
    return C

def track(frames, robot_free_idx=None, log=None):
    seg = Seg(frames)
    seg.calibrate(frames[:5])
    C, med = init_tracks(seg, frames[:5])
    N = len(C)
    traj = [C.copy()]; vis = [np.ones(N, bool)]
    V = np.zeros_like(C)
    for f in frames[1:]:
        m = seg.mask(f)
        ys, xs = np.nonzero(m); P = np.stack([ys, xs], 1).astype(np.float32)
        pred = C + 0.5 * V
        if len(P) < 3:
            traj.append(C.copy()); vis.append(np.zeros(N, bool)); continue
        Cn = kmeans(P, pred, iters=6)
        d = ((P[:, None, :] - Cn[None]) ** 2).sum(-1)
        cnt = np.bincount(d.argmin(1)[d.min(1) < 14 ** 2], minlength=N)
        v = cnt >= 3
        Cn[~v] = C[~v]
        V = 0.6 * V + 0.4 * (Cn - C); V[~v] = 0
        C = Cn
        traj.append(C.copy()); vis.append(v)
    return np.array(traj), np.array(vis), seg

def picked_index(traj, vis):
    T, N, _ = traj.shape
    disp = np.linalg.norm(traj - traj[0][None], axis=-1)  # T,N
    moving = (disp > 4).sum(1)
    ts = T
    for t in range(T):
        if moving[t] >= 3:
            ts = t; break
    score = disp[:ts].max(0) + 0.3 * (~vis[:ts]).sum(0)
    return int(score.argmax()), ts, score


def detect(seg, f, min_area=10):
    m = seg.mask(f)
    cents, areas = count_objects(m, min_area)
    return np.array(cents, np.float32).reshape(-1, 2), np.array(areas), m


def identify(frames, times, pre_t=1.0, post_t=10.4, log=print, hist=None):
    """frames: front RGB arrays of the conditioning video (+ first execution frame at end);
    times: their source timestamps (s). Returns dict with final target pixel (row, col)."""
    from scipy.optimize import linear_sum_assignment
    times = np.asarray(times, float)
    pre_idx = [i for i in range(len(frames)) if times[i] <= pre_t + 1e-6] or [0]
    post_i = int(np.argmin(np.abs(times - post_t)))
    seg = Seg(); seg.calibrate([frames[i] for i in pre_idx[:3]])
    seg.build_exclusion([frames[i] for i in range(post_i, len(frames), 8)])
    C0, med = init_tracks(seg, [frames[i] for i in pre_idx[:4]])
    Cp, _, _ = detect(seg, frames[post_i])
    N = len(C0)
    D = np.linalg.norm(C0[:, None] - Cp[None], axis=-1) if len(Cp) else np.zeros((N, 0))
    r, c = linear_sum_assignment(D) if len(Cp) else ([], [])
    disp = np.full(N, np.inf)
    C = C0.copy()
    for i, j in zip(r, c):
        disp[i] = D[i, j]; C[i] = Cp[j]
    # unmatched C0 (np.inf) -> picked object vanished (e.g. placed behind an occluder)
    order = np.argsort(-np.where(np.isfinite(disp), disp, 1e3))
    k = int(order[0])
    second = disp[order[1]] if N > 1 else 0.0
    conf = float(min(disp[k], 50) - (second if np.isfinite(second) else 50))
    log(f"[track] N={N} hue={seg.obj_hue} post_i={post_i} disp={np.round(disp,1).tolist()} picked={k} conf={conf:.1f}")
    if not np.isfinite(disp[k]):
        C[k] = C0[k]
    # k-means tracking from post frame to the end
    V = np.zeros_like(C); vis_hist = []
    lost = np.zeros(N, int)
    for f in frames[post_i + 1:]:
        m = seg.mask(f)
        ys, xs = np.nonzero(m); P = np.stack([ys, xs], 1).astype(np.float32)
        if len(P) < 3:
            continue
        pred = C + 0.5 * V
        Cn = kmeans(P, pred, iters=6)
        d = ((P[:, None, :] - Cn[None]) ** 2).sum(-1)
        cnt = np.bincount(d.argmin(1)[d.min(1) < 14 ** 2], minlength=N)
        v = cnt >= 3
        Cn[~v] = C[~v]
        V = 0.6 * V + 0.4 * (Cn - C); V[~v] = 0
        lost[~v] += 1
        C = Cn
        if hist is not None: hist.append(C.copy())
    log(f"[track] final target rc={C[k].round(1).tolist()} lost_frames={lost[k]} all={C.round(0).astype(int).tolist()}")
    return {"target_rc": C[k].tolist(), "all_rc": C.tolist(), "picked": k, "conf": conf, "hue": seg.obj_hue,
            "disp": disp.tolist(), "lost": int(lost[k]), "N": N}
