"""Pure decision logic: instruction parsing, visit counting, press trigger."""
import re

import numpy as np

ORD = {
    'first': 1, 'once': 1, 'one': 1,
    'second': 2, 'twice': 2, 'two': 2,
    'third': 3, 'thrice': 3, 'three': 3,
    'fourth': 4, 'four': 4, 'fifth': 5, 'five': 5, 'sixth': 6, 'six': 6,
    'seventh': 7, 'seven': 7, 'eighth': 8, 'eight': 8, 'ninth': 9, 'nine': 9,
    'tenth': 10, 'ten': 10,
}


def parse_count(instr):
    """Visit index N named by the instruction. Returns (N, how)."""
    s = (instr or '').lower()
    m = re.search(r'\b(\d+)\s*(st|nd|rd|th)\b', s)
    if m:
        return int(m.group(1)), 'ordinal_digit'
    m = re.search(r'\b(\d+)\s*(times|time|x)\b', s)
    if m:
        return int(m.group(1)), 'digit_times'
    # word forms, prefer ones adjacent to 'time'
    m = re.search(r'\b(' + '|'.join(ORD) + r')\s+(times?)\b', s)
    if m:
        return ORD[m.group(1)], 'word_times'
    m = re.search(r'\bfor the\s+(' + '|'.join(ORD) + r')\b', s)
    if m:
        return ORD[m.group(1)], 'for_the_word'
    for w in ('first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
              'once', 'twice', 'thrice'):
        if re.search(r'\b' + w + r'\b', s):
            return ORD[w], 'word'
    m = re.search(r'\b(\d+)\b', s)
    if m and 1 <= int(m.group(1)) <= 10:
        return int(m.group(1)), 'bare_digit'
    for w in ('two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'):
        if re.search(r'\b' + w + r'\b', s):
            return ORD[w], 'bare_word'
    return 1, 'default'


def seg_point_dist(a, b, p):
    a, b, p = np.asarray(a, float), np.asarray(b, float), np.asarray(p, float)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return float(np.linalg.norm(p - a))
    s = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return float(np.linalg.norm(a + s * ab - p))


def count_visits(track, target_xy, r_in, hyst=0.006):
    """track: list of (t, xy). A visit = cube centre comes within r_in of the
    target and later leaves beyond r_in+hyst; a jump between two samples whose
    segment passes within r_in counts as an entered-and-left visit.
    A visit already in progress at the first sample is not counted.
    Returns dict(completed, inside, initial_inside, events)."""
    T = np.asarray(target_xy, float)
    completed = 0
    inside = False
    initial_inside = False
    events = []
    prev = None
    for i, (t, xy) in enumerate(track):
        xy = np.asarray(xy, float)
        d = float(np.linalg.norm(xy - T))
        if prev is None:
            inside = d < r_in
            initial_inside = inside
        elif not inside:
            if d < r_in:
                inside = True
                events.append(('enter', t))
            elif seg_point_dist(prev[1], xy, T) < r_in:
                completed += 1
                events.append(('skip', t))
        else:
            if d > r_in + hyst:
                inside = False
                if initial_inside:
                    initial_inside = False
                    events.append(('exit_initial', t))
                else:
                    completed += 1
                    events.append(('exit', t))
        prev = (t, xy)
    return {'completed': completed, 'inside': inside, 'initial_inside': initial_inside, 'events': events}


def fit_motion(track, now_t, window=6, min_pts=2):
    """Least-squares constant-velocity fit on the last samples within window steps.
    Returns (p_now, v_per_step, n) or None."""
    pts = [(t, np.asarray(xy, float)) for t, xy in track if t >= now_t - window]
    if len(pts) < min_pts:
        return None
    ts = np.array([p[0] for p in pts], float)
    X = np.array([p[1] for p in pts])
    if np.ptp(ts) < 1e-9:
        return X[-1], np.zeros(2), len(pts)
    A = np.stack([np.ones_like(ts), ts - now_t], 1)
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    return coef[0], coef[1], len(pts)


def press_decision(track, target_xy, r_in, N, now_t, L, last_seen_t):
    """Whether to start the press now. L = steps between issuing the press and the
    button registering. Returns (bool, info)."""
    info = {}
    cv = count_visits(track, target_xy, r_in)
    info['completed'] = cv['completed']
    info['inside'] = cv['inside']
    info['initial_inside'] = cv['initial_inside']
    if cv['completed'] != N - 1:
        return False, info
    if cv['initial_inside'] and cv['inside']:
        # a visit in progress at the first sample is not a counted arrival
        return False, info
    if last_seen_t is None or now_t - last_seen_t > 3:
        info['why'] = 'stale'
        return False, info
    fm = fit_motion(track, now_t)
    if fm is None:
        return False, info
    p, v, n = fm
    T = np.asarray(target_xy, float)
    d_now = float(np.linalg.norm(p - T))
    sp = float(np.linalg.norm(v))
    info.update(d_now=round(d_now, 4), speed=round(sp, 5), n=n)
    if sp < 3e-4:
        ok = cv['inside'] and d_now < r_in
        info['why'] = 'stationary_inside' if ok else 'stationary'
        return ok, info
    tstar = float(-(p - T) @ v / (sp * sp))
    dmin = float(np.linalg.norm(p + v * tstar - T))
    info.update(tstar=round(tstar, 2), dmin=round(dmin, 4))
    if tstar >= 0:
        if tstar <= L + 0.5 and dmin < r_in:
            info['why'] = 'predicted_closest'
            return True, info
        return False, info
    # already past the closest point: late press while still well inside
    if cv['inside'] and d_now < 0.6 * r_in:
        info['why'] = 'late_inside'
        return True, info
    return False, info


def steps_to_event(track, target_xy, now_t):
    """Rough number of steps until the cube's closest approach (inf if unknown)."""
    fm = fit_motion(track, now_t, window=8)
    if fm is None:
        return float('inf')
    p, v, _ = fm
    sp = float(np.linalg.norm(v))
    if sp < 3e-4:
        return float('inf')
    tstar = float(-(p - np.asarray(target_xy, float)) @ v / (sp * sp))
    return tstar if tstar > 0 else float('inf')
