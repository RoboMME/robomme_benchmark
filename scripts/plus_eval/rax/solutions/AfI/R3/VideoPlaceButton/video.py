"""Parse the conditioning video (EE state + gripper width) into grasp / release / press events."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

HOLD_MIN, HOLD_MAX, OPEN_MIN, PRESS_MAX = 0.015, 0.068, 0.072, 0.010


def load_manifest(manifest_path, root="/workspace"):
    m = json.loads((Path(root) / manifest_path).read_text())
    frames = sorted(m["frames"], key=lambda f: f["frame_index"])
    return m, frames


def parse_events(frames):
    vid = [f for f in frames if f.get("phase") == "video_demo"]
    if not vid:
        return [], vid
    S = np.array([f["state"] for f in vid], float)
    w = S[:, 6] + S[:, 7]
    z = S[:, 2]
    state = []  # per-frame gripper class
    for wi in w:
        if wi >= OPEN_MIN:
            state.append("open")
        elif wi <= PRESS_MAX:
            state.append("closed_empty")
        elif HOLD_MIN <= wi <= HOLD_MAX:
            state.append("hold")
        else:
            state.append("mid")
    events = []
    # Stable-class segmentation (ignore 'mid' transients)
    prev = None; prev_i = 0
    for i, s in enumerate(state):
        if s == "mid":
            continue
        if prev is not None and s != prev:
            if prev == "open" and s == "hold":
                events.append({"type": "grasp", "i": i, "pos": S[i, :3].tolist(), "rot": S[i, 3:6].tolist()})
            elif prev == "hold" and s == "open":
                # release position: last hold frame before opening
                j = i - 1
                while j > prev_i and state[j] != "hold":
                    j -= 1
                events.append({"type": "release", "i": i, "pos": S[j, :3].tolist(), "rot": S[j, 3:6].tolist()})
        prev = s; prev_i = i
    # Button presses: runs of closed_empty frames; press = lowest z within the run
    i = 0
    n = len(state)
    while i < n:
        if state[i] == "closed_empty":
            j = i
            while j < n and state[j] in ("closed_empty", "mid"):
                j += 1
            seg = np.arange(i, j)
            k = seg[np.argmin(z[seg])]
            if z[seg].max() - z[k] > 0.03:  # it actually descended
                events.append({"type": "press", "i": int(k), "pos": S[k, :3].tolist()})
            i = j
        else:
            i += 1
    events = [e for e in events if e["type"] == "press" or e["pos"][2] < 0.08]
    events.sort(key=lambda e: e["i"])
    for e in events:
        e["t"] = vid[e["i"]]["timestamp_s"]
        e["frame_index"] = vid[e["i"]]["frame_index"]
    return events, vid


def held_color(frames_by_index, frame_index, root="/workspace", colors=("red", "green", "blue", "yellow")):
    """Colour occupying the wrist image's lower centre (the held cube) at a video frame."""
    from perception import mask
    f = frames_by_index.get(frame_index)
    if f is None:
        return None, {}
    files = {(x["camera"], x["kind"]): x["path"] for x in f["files"]}
    p = files.get(("wrist", "rgb"))
    if p is None:
        return None, {}
    rgb = np.array(Image.open(Path(root) / p).convert("RGB"))
    roi = rgb[150:256, 70:186]
    frac = {c: float(mask(roi, c).mean()) for c in colors}
    best = max(frac, key=frac.get)
    return (best if frac[best] > 0.05 else None), frac
