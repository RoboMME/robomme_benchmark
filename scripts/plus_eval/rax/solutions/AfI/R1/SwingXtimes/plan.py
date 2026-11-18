"""Instruction parsing and visit-sequence planning (pure functions, unit-tested offline)."""
import re

from percept import COLOR_HUES, ACHROMATIC

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "a": 1, "single": 1}
ADVERB = {"once": 1, "twice": 2, "thrice": 3}


def parse_instruction(text):
    t = (text or "").lower()
    out = {"text": t, "color": None, "count": None, "count_src": None, "first": None,
           "button": "button" in t, "final_side": None, "more": False}
    words = re.findall(r"[a-z]+|\d+", t)
    allc = set(COLOR_HUES) | ACHROMATIC
    # colour: the colour word nearest before 'cube'/'block'
    for i, w in enumerate(words):
        if w in ("cube", "block", "box") and i > 0:
            for j in range(i - 1, max(-1, i - 4), -1):
                if words[j] in allc:
                    out["color"] = words[j]
                    break
            if out["color"]:
                break
    if out["color"] is None:
        for w in words:
            if w in allc:
                out["color"] = w
                break
    # count
    m = re.search(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a single)\s+(more\s+|additional\s+|full\s+|complete\s+)?(times|swings|rounds|cycles|repetitions)", t)
    if m:
        g = m.group(1).replace("a single", "single")
        n = int(g) if g.isdigit() else NUM_WORDS.get(g)
        out["count"], out["count_src"] = n, m.group(0)
        if m.group(2) and ("more" in m.group(2) or "additional" in m.group(2)):
            out["more"] = True
            out["count"] = n + 1
    else:
        for k, v in ADVERB.items():
            if re.search(r"\b%s\b" % k, t):
                out["count"], out["count_src"] = v, k
                break
    # first-named side
    ir = [i for i in (t.find("right"),) if i >= 0]
    il = [i for i in (t.find("left"),) if i >= 0]
    if ir and il:
        out["first"] = "right" if ir[0] < il[0] else "left"
    elif ir:
        out["first"] = "right"
    elif il:
        out["first"] = "left"
    # final placement on a named target?
    m2 = None
    for m2 in re.finditer(r"(put|place|set|lower)\b[^,.;]*?\b(down|on|onto)\b[^,.;]*", t):
        pass
    if m2 is not None:
        seg = m2.group(0)
        if "target" in seg:
            if "right" in seg:
                out["final_side"] = "right"
            elif "left" in seg:
                out["final_side"] = "left"
    return out


def visit_sequence(first, count, ambiguous):
    """List of side labels to visit. first: 'right'/'left'/None. ambiguous: the left/right
    naming convention could be either (camera faces the robot), so use 2X+1 alternating
    visits starting with the second-named side, which completes X first->second swings
    under both naming conventions (see memory/development_report.md)."""
    count = max(1, min(int(count or 1), 12))
    if first is None:
        first = "right"
    second = "left" if first == "right" else "right"
    if ambiguous:
        seq = []
        for _ in range(count):
            seq += [second, first]
        seq.append(second)
        return seq
    seq = []
    for _ in range(count):
        seq += [first, second]
    return seq


def count_from_video(manifest, log=print):
    """Fallback when the instruction has no number: count lateral swings of the robot hand in
    the conditioning video's per-frame states (only if the video shows the robot moving)."""
    try:
        if not isinstance(manifest, dict):
            return None
        frames = [f for f in manifest.get("frames", []) if isinstance(f, dict) and f.get("phase") == "video_demo"]
        frames.sort(key=lambda f: f.get("frame_index", 0))
        pts = []
        for f in frames:
            s = f.get("state")
            if isinstance(s, list) and len(s) >= 3:
                pts.append([float(x) for x in s[:3]])
        if len(pts) < 10:
            return None
        import numpy as np
        P = np.array(pts)
        xy = P[:, :2] - P[:, :2].mean(0)
        if np.ptp(xy, axis=0).max() < 0.08:
            return None
        w, v = np.linalg.eigh(np.cov(xy.T))
        axis = v[:, -1]
        s = xy @ axis
        lo, hi = np.percentile(s, 10), np.percentile(s, 90)
        if hi - lo < 0.08:
            return None
        th_lo, th_hi = lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo)
        ext, state = [], None
        for x in s:
            if x > th_hi and state != "hi":
                state = "hi"
                ext.append("hi")
            elif x < th_lo and state != "lo":
                state = "lo"
                ext.append("lo")
        n = len(ext) // 2
        log("video swing extremes: %d -> count %d" % (len(ext), n))
        return n if n >= 1 else None
    except Exception as e:
        log("count_from_video failed:", repr(e))
        return None
