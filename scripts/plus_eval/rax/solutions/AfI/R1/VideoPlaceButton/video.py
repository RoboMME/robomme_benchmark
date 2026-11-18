"""Conditioning-video analysis from per-frame robot states, and instruction parsing."""
import math
import re
import numpy as np

from io_utils import log
from geom import rotvec_to_mat

PROMINENCE = 0.025   # m, minimum dip depth of a contact (place / grasp / press)
GRIP_DELTA = 0.010   # m, finger-sum change that counts as open / close
LOW_BAND = 0.015     # m, samples this close to the dip minimum form its low phase


def video_frames(manifest):
    frames = (manifest or {}).get("frames", []) or []
    vid = [f for f in frames if str(f.get("phase", "")).lower() in ("video_demo", "video", "demo", "conditioning")]
    vid.sort(key=lambda f: (f.get("frame_index", 0)))
    ex = [f for f in frames if str(f.get("phase", "")).lower() == "execution"]
    return vid, ex


def states_of(frames):
    S, T = [], []
    for k, f in enumerate(frames):
        s = f.get("state")
        if not (isinstance(s, list) and len(s) >= 8):
            continue
        try:
            S.append([float(x) for x in s[:8]])
            t = f.get("timestamp_s")
            T.append(float(t) if t is not None else float(f.get("frame_index", k)) / 20.0)
        except Exception:
            continue
    return np.asarray(S), np.asarray(T)


def _prominent_minima(z, prom):
    n = len(z)
    mins = []
    i = 0
    # plateau-aware local minima
    while i < n:
        j = i
        while j + 1 < n and abs(z[j + 1] - z[i]) < 1e-6:
            j += 1
        left = z[i - 1] if i > 0 else np.inf
        right = z[j + 1] if j + 1 < n else np.inf
        if z[i] < left and z[i] < right:
            mins.append((i + j) // 2)
        i = j + 1
    out = []
    for m in mins:
        # left prominence
        k = m
        lmax = z[m]
        while k > 0:
            k -= 1
            lmax = max(lmax, z[k])
            if z[k] < z[m] - 1e-6:
                break
        if k == 0 and z[0] >= z[m]:
            lmax = max(lmax, z[0])
        k = m
        rmax = z[m]
        while k < n - 1:
            k += 1
            rmax = max(rmax, z[k])
            if z[k] < z[m] - 1e-6:
                break
        p = min(lmax, rmax) - z[m]
        # an episode boundary counts as high if the arm starts/ends high
        if m == 0 or m == n - 1:
            continue
        if p >= prom:
            out.append(m)
    # merge minima inside one low phase
    merged = []
    for m in out:
        if merged and np.max(z[merged[-1]:m + 1]) - max(z[m], z[merged[-1]]) < prom:
            if z[m] < z[merged[-1]]:
                merged[-1] = m
            continue
        merged.append(m)
    return merged


def detect_events(S, T, table_z=None):
    """Return list of events {type: place|grasp|press, t, xy, z, yaw, g_before, g_after}."""
    if len(S) < 5:
        return [], {"reason": "too few states"}
    pos = S[:, :3]
    g = S[:, 6] + S[:, 7]
    z = pos[:, 2]
    info = {"n": int(len(S)), "z_min": float(z.min()), "z_max": float(z.max()),
            "g_min": float(g.min()), "g_max": float(g.max()),
            "pos_range": float(np.linalg.norm(pos.max(0) - pos.min(0)))}
    if info["pos_range"] < 0.02 and g.max() - g.min() < 0.005:
        info["reason"] = "static states"
        return [], info
    mins = _prominent_minima(z, PROMINENCE)
    events = []
    for m in mins:
        zm = z[m]
        a = m
        while a > 0 and z[a - 1] < zm + LOW_BAND:
            a -= 1
        b = m
        while b < len(z) - 1 and z[b + 1] < zm + LOW_BAND:
            b += 1
        a0 = max(a - 2, 0)
        b0 = min(b + 2, len(z) - 1)
        g_in = float(g[a0])
        g_out = float(g[b0])
        low = slice(a, b + 1)
        xy = pos[low, :2].mean(0)
        if table_z is not None and zm > table_z + 0.30:
            continue
        dg = g_out - g_in
        if dg > GRIP_DELTA:
            typ = "place"
        elif dg < -GRIP_DELTA:
            typ = "grasp"
        else:
            typ = "press"
        try:
            R = rotvec_to_mat(S[m, 3:6])
            yaw = math.atan2(R[1, 0], R[0, 0])
        except Exception:
            yaw = None
        events.append({"type": typ, "i": int(m), "t": float(T[m]), "t0": float(T[a]), "t1": float(T[b]),
                       "xy": xy, "z": float(zm), "yaw": yaw, "g_in": g_in, "g_out": g_out,
                       "g_low_mean": float(g[low].mean())})
    # spurious dips: drop presses at a grasp/place location (keep at least one press)
    presses = [e for e in events if e["type"] == "press"]
    contacts = [e for e in events if e["type"] != "press"]
    if len(presses) > 1 or (presses and contacts):
        keep = [p for p in presses if all(np.linalg.norm(p["xy"] - c["xy"]) > 0.03 for c in contacts)]
        if keep and len(keep) < len(presses):
            info["dropped_presses"] = [(round(p["t"], 2), np.round(p["xy"], 3).tolist()) for p in presses if p not in keep]
            events = [e for e in events if e["type"] != "press" or e in keep]
            presses = keep
    if len(presses) > 1:
        # several presses at different places: the button is where the lowest-energy cluster is
        groups = []
        for p in presses:
            for g_ in groups:
                if np.linalg.norm(g_[0]["xy"] - p["xy"]) < 0.03:
                    g_.append(p)
                    break
            else:
                groups.append([p])
        if len(groups) > 1:
            best = max(groups, key=lambda g_: (len(g_), -min(q["z"] for q in g_)))
            info["press_groups"] = [[(round(q["t"], 2), np.round(q["xy"], 3).tolist()) for q in g_] for g_ in groups]
            events = [e for e in events if e["type"] != "press" or any(e is q for q in best)]
    info["n_events"] = len(events)
    return events, info


def cube_history(events):
    """Resting places of the cube: [(t, xy, z, kind)] where kind initial|place."""
    hist = []
    grasps = [e for e in events if e["type"] == "grasp"]
    places = [e for e in events if e["type"] == "place"]
    if grasps and (not places or grasps[0]["t"] < places[0]["t"]):
        hist.append((-1e9, grasps[0]["xy"], grasps[0]["z"], "initial"))
    for p in places:
        hist.append((p["t"], p["xy"], p["z"], "place"))
    return hist


# ---------------------------------------------------------------- instruction
ORD = {"first": 1, "1st": 1, "once": 1, "second": 2, "2nd": 2, "twice": 2, "third": 3, "3rd": 3,
       "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6}


def parse_instruction(text):
    s = (text or "").lower()
    words = re.findall(r"[a-z0-9']+", s)
    res = {"text": text, "anchor": None, "k": 1, "press_k": 1, "use_last": False, "do_press": False,
           "press_first": False}
    # anchor relative to the button press
    idx_btn = [i for i, w in enumerate(words) if w.startswith("button") or w.startswith("press")]
    cand = []
    for i, w in enumerate(words):
        a = None
        if w in ("before", "prior", "preceding", "earlier"):
            a = "before"
        elif w in ("after", "following", "afterwards", "afterward", "subsequent", "subsequently", "later", "next"):
            a = "after"
        elif w in ("when", "while", "during"):
            a = "during"
        if a:
            dist = min([abs(i - j) for j in idx_btn]) if idx_btn else 99
            cand.append((dist, 0 if a != "during" else 1, i, a))
    if cand:
        cand.sort()
        res["anchor"] = cand[0][3]
        ai = cand[0][2]
        # ordinal for the placement: within 6 words before the anchor
        for w in words[max(0, ai - 10):ai]:
            if w in ORD:
                res["k"] = ORD[w]
            if w == "last":
                res["use_last"] = True
        # ordinal for the press: "<ord> time" / "<ord> press" after a press word
        for j in idx_btn:
            win = words[j:j + 6]
            for q, w in enumerate(win):
                if w in ORD and ((q + 1 < len(win) and win[q + 1].startswith("time")) or j > ai):
                    res["press_k"] = ORD[w]
    else:
        for w in words:
            if w in ORD:
                res["k"] = ORD[w]
            if w in ("last", "final", "latest", "most"):
                res["use_last"] = True
    # imperative press in the current execution?
    if re.search(r"(^|[.,;:]\s*|\bthen\s+|\band\s+|\bfirst\s+|\bfinally\s+)press\s+the\s+button", s):
        res["do_press"] = True
        pi = s.find("press the button")
        pl = s.find("place")
        res["press_first"] = pl < 0 or pi < pl
    return res


def choose_target(plan, events, hist):
    """Return (event-like dict of chosen resting place, why)."""
    places = [e for e in events if e["type"] == "place"]
    presses = [e for e in events if e["type"] == "press"]
    if not places and not hist:
        return None, "no placements in video"
    anchor = plan.get("anchor")
    k = max(1, int(plan.get("k", 1)))
    if anchor in ("before", "during", "after") and presses:
        pk = min(max(1, plan.get("press_k", 1)), len(presses)) - 1
        tp = presses[pk]["t"]
        if anchor in ("before", "during"):
            prev = [h for h in hist if h[0] < tp]
            if prev:
                h = prev[-min(k, len(prev))]
                return {"xy": h[1], "z": h[2], "t": h[0], "kind": h[3]}, "%s press@%.2f k=%d" % (anchor, tp, k)
            if places:
                e = places[0]
                return {"xy": e["xy"], "z": e["z"], "t": e["t"], "kind": "place"}, "before-none-fallback-first"
        else:
            nxt = [e for e in places if e["t"] > tp]
            if nxt:
                e = nxt[min(k, len(nxt)) - 1]
                return {"xy": e["xy"], "z": e["z"], "t": e["t"], "kind": "place"}, "after press@%.2f k=%d" % (tp, k)
            e = places[-1]
            return {"xy": e["xy"], "z": e["z"], "t": e["t"], "kind": "place"}, "after-none-fallback-last"
    # no anchor / no press detected
    if not places:
        h = hist[-1]
        return {"xy": h[1], "z": h[2], "t": h[0], "kind": h[3]}, "hist-last"
    if plan.get("use_last"):
        e = places[-1]
        return {"xy": e["xy"], "z": e["z"], "t": e["t"], "kind": "place"}, "last placement"
    if anchor == "after":
        e = places[-1]
        why = "after but no press: last placement"
    elif anchor in ("before", "during"):
        e = places[0] if len(places) < 3 else places[len(places) // 2 - 1]
        why = "before but no press: early placement"
    else:
        e = places[min(k, len(places)) - 1]
        why = "ordinal k=%d" % k
    return {"xy": e["xy"], "z": e["z"], "t": e["t"], "kind": "place"}, why


# ---------------------------------------------------------------- image-only fallback
def image_history(frames_v, front_map, tz, cube, button, load_frame, max_frames=160, cube_h=0.04):
    """Fallback when per-frame states are unusable: track the cube by colour in the front
    camera and detect the press as arm occupancy above the button.
    Returns (events, hist, info)."""
    info = {}
    if front_map is None or cube is None or not frames_v:
        return [], [], {"reason": "missing inputs"}
    ccol = np.asarray(cube.get("color") or [0, 0, 0], dtype=float)
    idx = np.unique(np.round(np.linspace(0, len(frames_v) - 1, min(max_frames, len(frames_v)))).astype(int))
    rec = []
    for i in idx:
        f = frames_v[i]
        try:
            rgb, depth = load_frame(f)
        except Exception:
            continue
        if rgb is None or depth is None:
            continue
        P = front_map.backproject(depth)
        h = P[..., 2] - tz
        valid = np.isfinite(h)
        cd = np.linalg.norm(rgb.astype(float) - ccol, axis=-1)
        m = valid & (cd < 35) & (np.nan_to_num(h, nan=-1) > 0.004)
        cpos = np.median(P[m], axis=0) if m.sum() >= 8 else None
        occ = 0
        if button is not None:
            bxy = np.asarray(button["xy"])
            near = valid & (np.linalg.norm(np.nan_to_num(P[..., :2], nan=9) - bxy, axis=-1) < 0.045)
            hh = np.nan_to_num(h, nan=-1)
            occ = int((near & (hh > button["top"] + 0.015) & (hh < button["top"] + 0.12)).sum())
        t = f.get("timestamp_s")
        t = float(t) if t is not None else float(f.get("frame_index", i)) / 20.0
        rec.append((t, cpos, occ))
    if not rec:
        return [], [], {"reason": "no frames"}
    # resting segments
    segs = []
    cur = None
    for t, c, _ in rec:
        rest = c is not None and (c[2] - tz) < cube_h + 0.012
        if rest and cur is not None and np.linalg.norm(c[:2] - cur["xy"]) < 0.012:
            cur["t1"] = t
            cur["pts"].append(c)
            continue
        if cur is not None:
            segs.append(cur)
            cur = None
        if rest:
            cur = {"t0": t, "t1": t, "xy": c[:2], "pts": [c]}
    if cur is not None:
        segs.append(cur)
    segs = [s for s in segs if len(s["pts"]) >= 2]
    merged = []
    for s in segs:
        xy = np.median(np.asarray(s["pts"])[:, :2], axis=0)
        if merged and np.linalg.norm(merged[-1]["xy"] - xy) < 0.02:
            merged[-1]["t1"] = s["t1"]
            continue
        merged.append({"t0": s["t0"], "t1": s["t1"], "xy": xy, "z": float(np.median(np.asarray(s["pts"])[:, 2]))})
    hist = []
    events = []
    for k, s in enumerate(merged):
        kind = "initial" if (k == 0 and s["t0"] <= rec[0][0] + 1e-6) else "place"
        zc = tz + 0.5 * cube_h + 0.003
        hist.append((s["t0"] if kind == "place" else -1e9, s["xy"], zc, kind))
        if kind == "place":
            events.append({"type": "place", "t": s["t0"], "xy": s["xy"], "z": zc, "yaw": None, "src": "image"})
    occ = np.asarray([r[2] for r in rec], dtype=float)
    info["occ_max"] = float(occ.max()) if len(occ) else 0
    if button is not None and len(occ) and occ.max() >= 6:
        k = int(np.argmax(occ))
        events.append({"type": "press", "t": rec[k][0], "xy": np.asarray(button["xy"]),
                       "z": tz + button["top"], "yaw": None, "src": "image"})
    events.sort(key=lambda e: e["t"])
    info["n_segments"] = len(merged)
    return events, hist, info
