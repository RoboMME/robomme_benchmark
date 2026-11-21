"""Read the reset conditioning sequence and identify the picked object in the final layout."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from robot import WS, log
from tracker import identify


def load_sequence(meta):
    seq = meta.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if not mp:
        return [], [], None
    man = json.loads((WS / mp).read_text())
    frames, times, phases = [], [], []
    for fr in sorted(man["frames"], key=lambda f: f["frame_index"]):
        files = {(f["camera"], f["kind"]): f["path"] for f in fr["files"]}
        if ("front", "rgb") not in files:
            continue
        with Image.open(WS / files[("front", "rgb")]) as im:
            frames.append(np.array(im.convert("RGB")))
        phases.append(fr["phase"])
        ts = fr.get("timestamp_s")
        times.append(float(ts) if ts is not None else float(fr.get("source_frame_index", 0)) / 20.0)
    vid = [i for i, p in enumerate(phases) if p == "video_demo"]
    exe = [i for i, p in enumerate(phases) if p != "video_demo"]
    log(f"[seq] {len(vid)} video frames, {len(exe)} exec frames; t=[{times[vid[0]] if vid else None},{times[vid[-1]] if vid else None}]")
    F = [frames[i] for i in vid] + ([frames[exe[0]]] if exe else [])
    T = [times[i] for i in vid] + ([times[vid[-1]] + 0.05] if exe else [])
    return F, T, man


def find_target(meta):
    F, T, man = load_sequence(meta)
    if len(F) < 10:
        return None
    return identify(F, T, log=log)
