"""Video analysis: which object was picked, and where it ended after the shuffle."""
import numpy as np
from scipy import ndimage
from vision import (rgb_to_hsv, mask_for_band, components, choose_object_hue,
                    adaptive_smin, hue_dist, PixelTracker, BlobTracker)


def object_mask(im, hue, smin):
    return mask_for_band(im, hue, width=25, smin=smin, vmin=40)


def static_exclusion(masks, init_idx=0):
    """Pixels that are object-coloured nearly always (tinted robot/scenery) plus init-frame
    components that are mostly static over the sequence."""
    mean = np.mean(np.stack(masks, 0).astype(np.float32), 0)
    st = mean > 0.85
    lab, n = ndimage.label(masks[init_idx])
    if n:
        idx = np.arange(1, n + 1)
        mm = ndimage.mean(mean, lab, idx)
        for i, v in zip(idx, mm):
            if v > 0.33:
                st |= lab == i
    return ndimage.binary_dilation(st, iterations=2)


def split_components(mask, minarea=4, a1=None):
    """Components; blobs much larger than a single object are split by k-means."""
    comps = components(mask, minarea=minarea)
    if not comps:
        return [], a1 or 0
    areas = np.array([c['area'] for c in comps])
    if a1 is None:
        big = areas[areas >= 6]
        a1 = float(np.median(big)) if len(big) else float(np.median(areas))
    lab, _ = ndimage.label(mask)
    out = []
    for c in comps:
        k = int(round(c['area'] / max(a1, 1)))
        if c['area'] < 0.3 * a1:
            continue
        if k >= 2 and c['area'] > 1.6 * a1:
            ys, xs = np.nonzero(lab == c['label'])
            P = np.stack([xs, ys], 1).astype(float)
            k = min(k, 4)
            # init along principal axis
            mu = P.mean(0)
            U, S, Vt = np.linalg.svd(P - mu, full_matrices=False)
            proj = (P - mu) @ Vt[0]
            qs = np.quantile(proj, (np.arange(k) + 0.5) / k)
            C = mu + qs[:, None] * Vt[0][None]
            for _ in range(10):
                a = np.argmin(np.linalg.norm(P[:, None] - C[None], axis=2), 1)
                C = np.array([P[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
            for j in range(k):
                out.append({'x': float(C[j, 0]), 'y': float(C[j, 1]), 'area': float((a == j).sum()), 'split': True})
        else:
            out.append({'x': c['x'], 'y': c['y'], 'area': c['area'], 'bbox': c.get('bbox')})
    return out, a1


def gray_mask(im):
    h, s, v = rgb_to_hsv(im)
    return (s < 70) & (v > 90)


def settle_from_images(fronts, times, prior=(6.0, 14.0, 9.8)):
    """Index where the robot stops after the initial pick/place (image fallback), with a
    plausibility window from the demos (robot stops ~9.6-10 s into the video)."""
    i = _settle_raw(fronts, times)
    lo, hi, mid = prior
    if not (lo <= times[i] - times[0] <= hi):
        i = int(np.argmin([abs(t - times[0] - mid) for t in times]))
    return i


def _settle_raw(fronts, times):
    n = len(fronts)
    if n < 4:
        return 0
    g = [gray_mask(im) for im in fronts]
    diff = np.array([0] + [int((g[i] ^ g[i - 1]).sum()) for i in range(1, n)], float)
    thr = max(60.0, 3.0 * float(np.median(diff[n // 2:])))
    moving = diff > thr
    if not moving.any():
        return 0
    first = int(np.argmax(moving))
    quiet_need = 1.5
    i = first
    while i < n:
        if not moving[i]:
            j = i
            while j < n and not moving[j]:
                j += 1
            if times[min(j, n - 1)] - times[i] >= quiet_need or j >= n:
                # require that significant motion occurred before (the pick)
                if times[i] - times[first] > 2.0:
                    return i
            i = j
        else:
            i += 1
    return 0


def state_events(states, times):
    """From per-frame states: release point of the first grasp and the settle index."""
    S = np.array(states, float)
    fs = S[:, 6] + S[:, 7]
    out = {}
    mx = float(np.max(fs))
    if mx < 0.01:
        return out
    closed = fs < 0.6 * mx
    n = len(S)
    i = 0
    grasp = None
    while i < n:
        if closed[i]:
            j = i
            while j < n and closed[j]:
                j += 1
            if i > 0 and times[min(j, n - 1)] - times[i] >= 0.3:
                grasp = (i, j)
                break
            i = j
        else:
            i += 1
    if grasp is None:
        return out
    gi, rj = grasp
    out['grasp_idx'] = gi
    out['grasp_pos'] = S[gi, :3].copy()
    out['holding_width'] = float(np.median(fs[gi:rj]))
    if rj >= n:
        out['release_idx'] = None
        return out
    out['release_idx'] = rj
    out['release_pos'] = S[max(rj - 1, gi), :3].copy()
    # settle: after release, first k where EE stays within 3 mm over the next 1.0 s
    P = S[:, :3]
    for k in range(rj, n):
        w = [q for q in range(k, n) if times[q] - times[k] <= 1.0]
        if len(w) < 2 or times[w[-1]] - times[k] < 0.8:
            break
        if np.max(np.linalg.norm(P[w] - P[k], axis=1)) < 0.003:
            out['settle_idx'] = k
            break
    if 'settle_idx' in out:
        k = out['settle_idx']
        # last index where the EE is still at the hold pose
        d = np.linalg.norm(P[k:] - P[k], axis=1)
        mv = np.nonzero(d > 0.005)[0]
        out['hold_end_idx'] = (k + int(mv[0]) - 1) if len(mv) else n - 1
        out['hold_state'] = S[k].copy()
    return out


def _object_like(dets, a1, shape=(256, 256)):
    out = []
    for d in dets:
        if not (0.35 * a1 <= d['area'] <= 2.5 * a1):
            continue
        bb = d.get('bbox')
        if bb is not None and (bb[0] <= 0 or bb[1] <= 0 or bb[2] >= shape[1] or bb[3] >= shape[0]):
            continue
        out.append(d)
    return out


def moved_object(img0, img_s, hue, smin, excl):
    """Image fallback: settle-frame detection whose frame-0 counterpart vanished."""
    m0 = object_mask(img0, hue, smin) & ~excl
    ms = object_mask(img_s, hue, smin) & ~excl
    d0, a1 = split_components(m0)
    ds, _ = split_components(ms, a1=a1 or None)
    d0 = _object_like(d0, a1, img0.shape)
    ds = _object_like(ds, a1, img0.shape)
    if not ds or not d0:
        return None, 0.0
    P0 = np.array([[d['x'], d['y']] for d in d0])
    Ps = np.array([[d['x'], d['y']] for d in ds])
    D = np.linalg.norm(Ps[:, None] - P0[None], axis=2)
    ds_s = D.min(1)   # settle det -> nearest frame0
    ds_0 = D.min(0)   # frame0 det -> nearest settle
    score = []
    for i in range(len(ds)):
        # vanished frame-0 objects within 60 px
        near = [ds_0[j] for j in range(len(d0)) if D[i, j] < 60]
        score.append(min(ds_s[i], max(near) if near else 0.0))
    order = np.argsort(score)[::-1]
    b = int(order[0])
    second = score[order[1]] if len(order) > 1 else 0.0
    conf = 0.0 if score[b] < 2.5 else min(1.0, (score[b] - second) / 6.0)
    return (ds[b]['x'], ds[b]['y']), conf


def track(images, times, hue, smin, init_idx, target_xy, excl=None, log=print, name='', kind='blob'):
    """Track all objects from images[init_idx] to the end; return info dict for target."""
    masks = [object_mask(im, hue, smin) for im in images[init_idx:]]
    if excl is None:
        excl = static_exclusion(masks)
    masks = [mk & ~excl for mk in masks]
    dets, a1 = split_components(masks[0])
    if len(dets) < 1 or a1 <= 0:
        return None
    P = np.array([[d['x'], d['y']] for d in dets])
    dist = np.linalg.norm(P - np.asarray(target_xy, float), axis=1)
    tgt = int(np.argmin(dist))
    if kind == 'blob':
        tr = BlobTracker(P, a1, init_areas=[d['area'] for d in dets])
    else:
        tr = PixelTracker(P, a1)
    T = times[init_idx:]
    lost_total = 0.0
    for k in range(1, len(masks)):
        tr.step(masks[k], T[k] - T[0])
        if tr.lost[tgt] > 0:
            lost_total += max(1e-3, T[k] - T[k - 1])
    H = np.array([h[tgt] for h in tr.history])
    info = {'n_tracks': len(dets), 'a1': a1, 'init_xy': P[tgt].tolist(), 'init_dist': float(dist[tgt]),
            'final_xy': H[-1].tolist(), 'lost_total': lost_total, 'lost_end': float(tr.lost[tgt]),
            'contested': float(tr.contested[tgt]), 'support_end': float(tr.support[tgt]),
            'history': H, 'tracker': tr, 'tgt': tgt}
    log('track[%s] n=%d a1=%.0f init=%s d=%.1f final=%s lost=%.2f lost_end=%.2f contested=%.2f' % (
        name, len(dets), a1, np.round(P[tgt], 1).tolist(), dist[tgt], np.round(H[-1], 1).tolist(),
        lost_total, tr.lost[tgt], tr.contested[tgt]))
    return info


def pick_hue(images):
    hue, scores = choose_object_hue([im for im in images if im is not None])
    return hue, scores
