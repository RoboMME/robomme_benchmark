"""Analyse the conditioning video's robot-state trajectory to find pick/place/press events."""
import json
from pathlib import Path
import numpy as np

WS = Path("/workspace")


def load_video_states(meta):
    seq = meta.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if not mp:
        return None, None
    man = json.loads((WS / mp).read_text())
    frames = [f for f in man["frames"] if f.get("phase") == "video_demo" and f.get("state")]
    frames.sort(key=lambda f: f["frame_index"])
    S = np.array([f["state"] for f in frames], float)
    return S, frames


def find_events(S, z_low=0.07):
    """Segment trajectory into low-z dips; classify each by gripper width before/after.
    Returns list of dicts: kind in {grasp, place, press, other}, xy, idx."""
    z = S[:, 2]
    g = S[:, 6] + S[:, 7]
    low = z < z_low
    events = []
    i, n = 0, len(S)
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j < n and low[j]:
            j += 1
        seg = slice(i, j)
        k = i + int(np.argmin(z[seg]))
        # gripper width just before descending and just after ascending
        g_before = g[max(0, i - 1)]
        g_after = g[min(n - 1, j)]
        g_min = g[seg].min()
        xy = S[k, :2].copy()
        if g_before > 0.065 and g_after < 0.06 and g_after > 0.01:
            kind = "grasp"
        elif g_before < 0.06 and g_before > 0.01 and g_after > 0.065:
            kind = "place"
            # release location: where gripper starts opening while low
            opening = [t for t in range(i, j) if g[t] > g_before + 0.01]
            if opening:
                xy = S[opening[0], :2].copy()
        elif g_before < 0.01 and g_min < 0.01:
            kind = "press"
        else:
            kind = "other"
        events.append(dict(kind=kind, xy=xy, idx=k, zmin=float(z[k]), start=i, end=j,
                           g_before=float(g_before), g_after=float(g_after)))
        i = j
    return events


def target_after_press(events):
    """xy of the first place event after the (first) button press."""
    presses = [e for e in events if e["kind"] == "press"]
    if not presses:
        return None, "no_press"
    t = presses[0]["idx"]
    places = [e for e in events if e["kind"] == "place" and e["idx"] > t]
    if not places:
        return None, "no_place_after_press"
    return places[0]["xy"], "ok"


def target_resting_at_press(events):
    """Hypothesis H1': the target the cube occupies when the button is pressed
    (last place before the press if the cube was resting), else the next place."""
    presses = [e for e in events if e["kind"] == "press"]
    if not presses:
        return None, "no_press"
    t = presses[0]["idx"]
    before = [e for e in events if e["kind"] in ("grasp", "place") and e["idx"] < t]
    if before and before[-1]["kind"] == "place":
        return before[-1]["xy"], "resting_at_press"
    places = [e for e in events if e["kind"] == "place" and e["idx"] > t]
    if places:
        return places[0]["xy"], "next_after_press"
    return None, "no_place"


def target_last_before_press(events):
    presses = [e for e in events if e["kind"] == "press"]
    if not presses:
        return None, "no_press"
    t = presses[0]["idx"]
    places = [e for e in events if e["kind"] == "place" and e["idx"] < t]
    if not places:
        return None, "no_place_before_press"
    return places[-1]["xy"], "ok_before"
