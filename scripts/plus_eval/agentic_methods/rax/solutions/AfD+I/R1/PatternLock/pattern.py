"""Decode the conditioning-video pattern: sequence of (row, col) cells on the video grid."""
import numpy as np
import vision

ON_MIN = 8          # red pixels in a cell ellipse to count as lit (tested on 3 demos: lit 41-92, unlit 0)
ON_OVER_BASE = 6    # must also exceed the cell's baseline (frame-0) red count by this

def _canonical_H():
    import json, os
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'canonical.json')) as f:
            return np.array(json.load(f)['H_canonical_front'], float)
    except Exception:
        return None

def grid_via_canonical(im, params=None):
    """Fallback for a non-axis-aligned video grid: rectify blobs with the fixed-camera canonical
    homography, fit the lattice, return image grid g[row, col]."""
    Hc = _canonical_H()
    if Hc is None:
        raise ValueError('no canonical H')
    bl = vision.merge_close(blobs_with(im, **(params or {})))
    pts = np.array([(u, v) for u, v, a in bl], float)
    G = vision.fit_grid_plane(vision.apply_h(Hc, pts))
    if G['count'] < 25 or G['rms'] > 0.08:
        raise ValueError('canonical fit weak: count %d rms %.3f' % (G['count'], G['rms']))
    planeW = vision.apply_h(Hc, pts)
    g = np.zeros((5, 5, 2))
    for (a, b), ks in G['members'].items():
        k = min(ks, key=lambda k: np.linalg.norm(planeW[k] - G['grid'][a, b]))
        g[b, a] = pts[k]
    return g

def detect_video_grid(front_frames, max_tries=12):
    """front_frames: list of RGB arrays (early video frames first). Returns (grid (5,5,2), frame_idx)."""
    last_err = None
    for k, im in enumerate(front_frames[:max_tries]):
        for params in ({}, {'diff_min': 30}, {'diff_min': 50}, {'sat_max': 70}, {'v_min': 70}):
            try:
                bl = vision.merge_close(blobs_with(im, **params))
                return vision.axis_aligned_grid(bl), k
            except Exception as e:  # try next params / frame
                last_err = e
    for k, im in enumerate(front_frames[:max_tries]):
        for params in ({}, {'diff_min': 30}, {'diff_min': 50}):
            try:
                return grid_via_canonical(im, params), k
            except Exception as e:
                last_err = e
    raise ValueError('video grid not found: %r' % (last_err,))

def blobs_with(im, **params):
    m = vision.target_mask(im, **params)
    out = []
    for ys, xs in vision.label(m):
        a = len(ys)
        if 8 <= a <= 220 and (ys.max() - ys.min()) < 19 and (xs.max() - xs.min()) < 23:
            out.append((xs.mean(), ys.mean(), a))
    return out

def cell_masks(centers_img, spacing_px):
    """centers_img: list of (u,v); spacing_px: list of local spacing. Returns list of bool masks."""
    yy, xx = np.mgrid[0:256, 0:256]
    out = []
    for (u, v), sp in zip(centers_img, spacing_px):
        out.append(((xx - u) ** 2 / (0.4 * sp) ** 2 + (yy - v) ** 2 / (0.3 * sp) ** 2) < 1)
    return out

def grid_masks(g):
    centers = []; sps = []
    for r in range(5):
        sp = float(np.mean(np.diff(g[r, :, 0])))
        for c in range(5):
            centers.append(tuple(g[r, c])); sps.append(sp)
    return cell_masks(centers, sps)

def red_counts(im, masks):
    R = vision.redness(im)
    return np.array([int(R[m].sum()) for m in masks])

def decode(front_frames, log=print):
    """Returns dict(sequence=[(r,c),...], events=[...], grid=g)."""
    g, k0 = detect_video_grid(front_frames)
    masks = grid_masks(g)
    T = np.array([red_counts(im, masks) for im in front_frames])
    base = np.median(T[:max(1, min(5, len(T)))], axis=0)
    on = (T >= ON_MIN) & (T >= base + ON_OVER_BASE)
    events = []
    prev = np.zeros(25, bool)
    for k in range(len(T)):
        new = np.nonzero(on[k] & ~prev)[0]
        # simultaneous onsets: order by red count (stronger first is arbitrary; log it)
        for c in sorted(new, key=lambda c: -T[k, c]):
            events.append((k, int(c // 5), int(c % 5), int(T[k, c])))
        prev = on[k]
    seq = []
    for k, r, c, cnt in events:
        if seq and seq[-1] == (r, c):
            continue  # flicker / occlusion re-onset of the same touch
        seq.append((r, c))
    log('video grid found at frame %d; events=%s' % (k0, events))
    return dict(sequence=seq, events=events, grid=g, counts=T)
