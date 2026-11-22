"""Parse the conditioning video (states per frame) into pick/place/press events and resolve the instruction."""
import json, re
import numpy as np
from pathlib import Path

WS = Path("/workspace")

ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4,
       "once": 1, "twice": 2, "thrice": 3}


def load_video_states(manifest_path):
    m = json.loads((WS / manifest_path).read_text())
    frames = [f for f in m["frames"] if f.get("phase") == "video_demo"]
    frames.sort(key=lambda f: f["frame_index"])
    S = np.array([f["state"] for f in frames], float)
    T = np.array([f.get("timestamp_s", i) for i, f in enumerate(frames)], float)
    return S, T, frames, m


def gripper_class(w):
    # w = sum of finger positions
    if w < 0.012:
        return "closed"   # empty and closed (button press)
    if w > 0.072:
        return "open"
    return "hold"         # holding something (cube ~0.037)


def extract_events(S, T=None):
    """Return list of events dicts: kind in {pick, place, press}, index, xy, z."""
    w = S[:, 6] + S[:, 7]
    cls = [gripper_class(x) for x in w]
    # smooth: require persistence of 3 frames
    n = len(cls)
    seg = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and cls[j + 1] == cls[i]:
            j += 1
        seg.append([cls[i], i, j])
        i = j + 1
    # merge short segments (<3 frames) that are transitional (e.g. closing passes through 'hold')
    merged = []
    for s in seg:
        if s[2] - s[1] + 1 < 3 and merged:
            merged[-1][2] = s[2]
        else:
            if merged and merged[-1][0] == s[0]:
                merged[-1][2] = s[2]
            else:
                merged.append(list(s))
    events = []
    prev = None
    for c, a, b in merged:
        zs = S[a:b + 1, 2]
        k = a + int(np.argmin(zs))
        if c == "hold":
            # pick happened at start of hold segment
            events.append(dict(kind="pick", index=a, xy=S[a, :2].copy(), z=float(S[a, 2])))
            # place at end of hold segment (if followed by open)
            events.append(dict(kind="place_pending", index=b, xy=S[b, :2].copy(), z=float(S[b, 2]), yaw_aa=S[b, 3:6].copy()))
        elif c == "closed":
            if zs.min() < 0.10:
                events.append(dict(kind="press", index=k, xy=S[k, :2].copy(), z=float(S[k, 2])))
    out = []
    for e in events:
        if e["kind"] == "place_pending":
            if e["index"] < n - 1:  # hold ended -> released
                e["kind"] = "place"
                out.append(e)
        else:
            out.append(e)
    return out


def parse_instruction(text):
    t = text.lower()
    color = None
    for c in ["red", "green", "blue", "yellow", "purple", "orange"]:
        if re.search(r"\b%s\b" % c, t):
            color = c
            break
    rel = "after" if "after" in t else ("before" if "before" in t else None)
    ordinal = 1
    m = re.search(r"\b(first|second|third|fourth|fifth|last|1st|2nd|3rd|4th)\b[^.]*\b(placed|put)", t)
    if m:
        ordinal = -1 if m.group(1) == "last" else ORD[m.group(1)]
    elif "last" in t or "right before" in t or "just before" in t:
        ordinal = -1 if rel == "before" else 1
    elif rel == "before":
        ordinal = -1
    # which press (if several)
    press_idx = 0
    m2 = re.search(r"button was pressed (?:for the )?(first|second|third|last)", t) or re.search(r"(first|second|third|last) (?:time )?(?:the )?button", t)
    if m2:
        press_idx = -1 if m2.group(1) == "last" else ORD[m2.group(1)] - 1
    return dict(color=color, rel=rel, ordinal=ordinal, press_idx=press_idx)


def resolve_target(events, spec):
    places = [e for e in events if e["kind"] == "place"]
    presses = [e for e in events if e["kind"] == "press"]
    if not places:
        return None, "no places"
    if not presses:
        return places[0], "no press found"
    p = presses[spec["press_idx"]] if -len(presses) <= spec["press_idx"] < len(presses) else presses[0]
    if spec["rel"] == "before":
        cand = [e for e in places if e["index"] < p["index"]]
        if not cand:
            return places[0], "no place before press"
        o = spec["ordinal"]
        return (cand[-1] if o == -1 else cand[-o] if o <= len(cand) else cand[0]), "before"
    cand = [e for e in places if e["index"] > p["index"]]
    if not cand:
        return places[-1], "no place after press"
    o = spec["ordinal"]
    if o == -1:
        return cand[-1], "after-last"
    return (cand[o - 1] if o <= len(cand) else cand[-1]), "after"


def assign_tracks(events):
    """Chain pick->place events into per-cube tracks. Adds 'track' to pick/place events."""
    tracks = []  # each: dict(pos=xy, start=xy)
    cur = None
    for e in events:
        if e["kind"] == "pick":
            best, bd = None, 0.035
            for i, t in enumerate(tracks):
                d = np.linalg.norm(t["pos"] - e["xy"])
                if d < bd:
                    best, bd = i, d
            if best is None:
                tracks.append(dict(pos=e["xy"].copy(), start=e["xy"].copy()))
                best = len(tracks) - 1
            cur = best
            e["track"] = cur
        elif e["kind"] == "place":
            if cur is not None:
                e["track"] = cur
                tracks[cur]["pos"] = e["xy"].copy()
                cur = None
    return tracks


def resolve_for_track(events, spec, track, semantics="naive"):
    if semantics == "flip_global":
        evs = list(events); semantics = "flip"
    else:
        evs = [e for e in events if e["kind"] == "press" or e.get("track") == track]
    sp = dict(spec)
    if semantics == "flip":
        if sp["rel"] == "after":
            sp["rel"] = "before"; sp["ordinal"] = -1 if sp["ordinal"] == 1 else sp["ordinal"]
        else:
            sp["rel"] = "after"; sp["ordinal"] = 1 if sp["ordinal"] == -1 else sp["ordinal"]
    return resolve_target(evs, sp)
