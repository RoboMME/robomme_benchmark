"""Classify the demonstrated manner from conditioning-video robot states."""
import json, os
import numpy as np


def load_video_states(meta, root="/workspace"):
    from demo import find_manifest
    path = find_manifest(meta, root)
    if not path:
        return None, "no manifest"
    m = json.load(open(path))
    frames = [f for f in m.get("frames", []) if f.get("phase") == "video_demo"]
    frames.sort(key=lambda f: f.get("frame_index", 0))
    S = [f.get("state") for f in frames]
    S = [s for s in S if s is not None and len(s) >= 8]
    if len(S) < 10:
        return None, f"only {len(S)} video states"
    return np.array(S, float), f"{len(S)} video states"


def classify(S):
    """Return (manner, info). manner in {'hook','pick','push_closed','push_open'}."""
    w = S[:, 6] + S[:, 7]
    z = S[:, 2]
    xy = S[:, :2]
    w_open = float(np.percentile(w[: max(3, len(w) // 20)], 50))
    closed = w < w_open - 0.012
    info = dict(w_open=w_open, w_min=float(w.min()), w_end=float(w[-1]), n_closed=int(closed.sum()))
    if closed.sum() < 3:
        return "push_open", info
    i0 = int(np.argmax(closed))
    # width while closed (median)
    wc = float(np.median(w[closed]))
    info["w_closed"] = wc
    # reopened after closing?
    reopen = (~closed[i0:]).any() and w[-1] > w_open - 0.012
    info["reopen"] = bool(reopen)
    # travel while closed and low
    low = closed & (z < 0.05)
    tr = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)[low[1:] & low[:-1]]))
    info["low_closed_travel"] = tr
    # lifted while closed
    info["z_max_closed"] = float(z[closed].max())
    if wc < 0.008:
        return "push_closed", info
    if reopen and tr < 0.15:
        return "pick", info
    return "hook", info
