"""Classify the conditioning video's manner from wrist-camera frames."""
import numpy as np
from perception import m_yellow, m_slate, m_salmon, m_gray, components

def m_teal2(a):
    a = a.astype(int); R, G, B = a[..., 0], a[..., 1], a[..., 2]
    return (G - R > 20) & (B - R > 18) & (G > 60) & (abs(G - B) < 35) & (B < 170)

def frame_class(w):
    roi = w[190:250, 150:190]
    fy, fs, ft = m_yellow(roi).mean(), m_slate(roi).mean(), m_teal2(roi).mean()
    best = max((fy, 'ball'), (fs, 'hammer'), (ft, 'teal'))
    return best[1] if best[0] > 0.35 else None

def shaft_ratio(w):
    t = m_teal2(w); t[:120] = False
    comps = components(t, 200)
    if not comps: return None
    a, ys, xs = comps[0]
    top = ys.min()
    rows = [(xs[ys == r].max() - xs[ys == r].min() + 1) for r in range(top + 3, top + 12) if (ys == r).any()]
    if not rows: return None
    tw = float(np.median(rows)); cx = xs[ys < top + 12].mean()
    if tw < 25: return None
    g = m_gray(w); ws = []
    for r in range(top - 25, top - 3):
        if r < 0: continue
        c = int(round(cx)); lo = max(0, c - 40)
        xs_ = np.where(g[r, lo:c + 40])[0] + lo
        if len(xs_) == 0: continue
        runs = np.split(xs_, np.where(np.diff(xs_) > 1)[0] + 1)
        ws.append(len(max(runs, key=len)))
    if len(ws) < 5: return None
    return float(np.median(ws)) / tw

def classify(wrist_frames):
    """wrist_frames: list of HxWx3 uint8 arrays (video order). Returns (label, info)."""
    counts = {'ball': 0, 'hammer': 0, 'teal': 0}
    teal_frames = []
    for w in wrist_frames:
        c = frame_class(w)
        if c: counts[c] += 1
        if c == 'teal': teal_frames.append(w)
    info = dict(counts=counts)
    n = len(wrist_frames)
    best = max(counts, key=counts.get)
    if counts[best] < max(3, 0.05 * n):
        return 'unknown', info
    if best != 'teal':
        return best, info
    sal = [m_salmon(w[0:120, 60:200]).mean() for w in teal_frames]
    info['plate_frac'] = float(np.median(sal))
    if np.median(sal) > 0.12:
        return 'plate', info
    rs = [r for r in (shaft_ratio(w) for w in teal_frames) if r is not None and r < 1.0]
    info['shaft_ratio'] = float(np.median(rs)) if rs else None
    if not rs:
        return 'stick', info
    return ('hook' if np.median(rs) > 0.33 else 'stick'), info
