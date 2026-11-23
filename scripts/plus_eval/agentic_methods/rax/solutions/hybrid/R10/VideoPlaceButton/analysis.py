"""Conditioning-video analysis: which target (slot at end of video) holds the answer."""
import numpy as np
from vision import detect_cube, detect_button, white_mask
from swaps import detect_swaps, follow

def wrist_white_cc(w):
    from scipy import ndimage
    a = w.astype(np.int32); R, G, B = a[..., 0], a[..., 1], a[..., 2]
    m = (R > 200) & (G > 200) & (B > 200) & (abs(R - G) < 12) & (abs(G - B) < 12) & (abs(R - B) < 12)
    lab, n = ndimage.label(m)
    return float(ndimage.sum(m, lab, range(1, n + 1)).max()) if n else 0.0

def find_press_wrist(wrists, times, log=print, thr=4800.0):
    v = np.array([wrist_white_cc(w) for w in wrists])
    k = int(np.argmax(v))
    log('wrist white cc max %.0f at idx %d t=%.2f (2nd group max below)' % (v[k], k, times[k]))
    if v[k] < thr:
        return None, v
    # first frame of the high group (press moment ~ start of contact)
    on = np.nonzero(v >= thr)[0]
    groups = np.split(on, np.nonzero(np.diff(on) > 3)[0] + 1)
    log('wrist press groups', [(int(g[0]), int(g[-1]), int(v[g].max())) for g in groups])
    g = max(groups, key=lambda g: v[g].max())
    return int(g[0]), v

def find_press(frames, times, cube_pts, log=print):
    # button = largest white blob in a frame with little arm; use median over a few frames
    btn = None
    for f in frames[:5] + frames[-3:]:
        b = detect_button(f)
        if b and (btn is None or b[2] > btn[2]):
            btn = b
    if btn is None:
        return None, None
    bmask = btn[3]
    vis = np.array([float((white_mask(f) & bmask).sum()) / max(1.0, bmask.sum()) for f in frames])
    low = np.nonzero(vis < 0.75)[0]
    if not len(low):
        return None, vis
    groups = np.split(low, np.nonzero(np.diff(low) > 3)[0] + 1)
    # candidate dips; prefer deepest dip, while cube not being carried near button
    best = min(groups, key=lambda g: vis[g].min())
    k = int(best[np.argmin(vis[best])])
    log('button at (%.0f,%.0f) area %d; dips %s; press idx %d t=%.2f vis=%.2f' % (
        btn[0], btn[1], btn[2], [(int(g[0]), int(g[-1]), round(float(vis[g].min()), 2)) for g in groups], k, times[k], vis[k]))
    return k, vis

def cube_rests(cube_pts, times, min_dur=0.8, tol=3.0):
    T = len(cube_pts)
    pts = np.array([[c[0], c[1]] if c else [np.nan, np.nan] for c in cube_pts])
    rests = []
    i = 0
    while i < T:
        if np.isnan(pts[i, 0]):
            i += 1; continue
        j = i
        while j + 1 < T and not np.isnan(pts[j + 1, 0]) and np.hypot(*(pts[j + 1] - pts[i])) < tol:
            j += 1
        if times[j] - times[i] >= min_dur:
            rests.append([i, j, float(np.median(pts[i:j + 1, 0])), float(np.median(pts[i:j + 1, 1]))])
        i = j + 1
    merged = []
    for r in rests:
        if merged and np.hypot(r[2] - merged[-1][2], r[3] - merged[-1][3]) < 6:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged

def analyze_video(frames, times, color, log=print, wrists=None):
    times = [float(t) for t in times]
    import vision
    vision.TARGET_MODE['mode'] = 'hue'
    slots, events, stat, dets = detect_swaps(frames, times, log=log)
    if len(slots) < 3:
        log('few slots (%d) with hue mask -> wide mask' % len(slots))
        vision.TARGET_MODE['mode'] = 'wide'
        slots, events, stat, dets = detect_swaps(frames, times, log=log)
    if len(slots) < 2:
        raise RuntimeError('no target slots found')
    log('slots', np.round(slots).tolist())
    cube = [detect_cube(f, color) for f in frames]
    press_f, vis = find_press(frames, times, cube, log=log)
    press_w = None
    if wrists is not None:
        press_w, _ = find_press_wrist(wrists, times, log=log)
    press = press_w if press_w is not None else press_f
    log('press front=%s wrist=%s -> %s' % (press_f, press_w, press))
    rests = cube_rests(cube, times)
    placements = []
    for (a, b, x, y) in rests:
        d = np.hypot(slots[:, 0] - x, slots[:, 1] - (y + 3))
        s = int(np.argmin(d))
        placements.append(dict(a=a, b=b, t=times[a], x=x, y=y, slot=s, dist=float(d[s]), on_target=bool(d[s] < 10)))
        log('rest t=%.2f-%.2f (%.0f,%.0f) slot %d d=%.1f' % (times[a], times[b], x, y, s, d[s]))
    res = dict(slots=slots, events=events, press=press, placements=placements)
    chosen = None
    if press is not None:
        for p in placements:
            if p['a'] > press and p['on_target']:
                chosen = p; break
    if chosen is None:
        # fallback: last on-target placement that is not the final rest position
        cands = [p for p in placements if p['on_target']]
        chosen = cands[-1] if cands else None
        log('WARNING fallback choice', chosen)
    res['chosen'] = chosen
    if chosen is not None:
        fs = follow(chosen['slot'], chosen['t'], events)
        res['final_slot'] = fs
        res['final_px'] = slots[fs].tolist()
        log('chosen placement slot %d at t=%.2f -> final slot %d px %s' % (chosen['slot'], chosen['t'], fs, np.round(slots[fs]).tolist()))
    return res
