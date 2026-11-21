"""Map the language instruction + video events to the referenced placement."""
import re

ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "1st": 1, "2nd": 2, "3rd": 3}
COLORS = ("red", "green", "blue", "yellow")


def parse(instr):
    s = instr.lower()
    color = next((c for c in COLORS if re.search(r"\b%s\b" % c, s)), "red")
    rel = "after"
    m = re.search(r"placed(.*?)(before|after)\s+the\s+button", s)
    if m:
        rel = m.group(2)
        tail = m.group(1)
    else:
        if "before" in s and "button" in s:
            rel = "before"
        tail = s
    # ordinal: search the phrase between 'where it was' and the relation word, else whole string
    seg = s
    m2 = re.search(r"where it was(.*?)(before|after)", s)
    if m2:
        seg = m2.group(1)
    ordinal = 1
    last = False
    explicit = False
    for w, k in ORD.items():
        if re.search(r"\b%s\b" % w, seg):
            ordinal = k
            explicit = True
            break
    if re.search(r"\b(last|final|lastly)\b", seg) or re.search(r"\b(right|just|immediately)\s+before\b", s):
        last = True
    if rel == "before" and not explicit:
        last = True  # "placed before the button was pressed" = the most recent placement before it
    return {"color": color, "relation": rel, "ordinal": ordinal, "last": last}


def select(events, q, held=None):
    """events: list from video.parse_events; held: optional dict frame_index->colour for releases."""
    presses = [e for e in events if e["type"] == "press"]
    rel = [e for e in events if e["type"] == "release"]
    if held:
        same = [e for e in rel if held.get(e["frame_index"]) in (q["color"], None)]
        if same:
            rel = same
    if not rel:
        return None, "no releases"
    if not presses:
        k = q["ordinal"] - 1
        return (rel[-1] if q["last"] else rel[min(k, len(rel) - 1)]), "no press found; ordinal over all"
    p = presses[0]["i"]
    if q["relation"] == "after":
        cand = [e for e in rel if e["i"] > p]
        if not cand:
            return rel[-1], "no release after press; using last"
        if q["last"]:
            return cand[-1], "last after press"
        return cand[min(q["ordinal"] - 1, len(cand) - 1)], "ordinal %d after press" % q["ordinal"]
    cand = [e for e in rel if e["i"] < p]
    if not cand:
        return rel[0], "no release before press; using first"
    if q["last"]:
        return cand[-1], "last before press"
    return cand[min(q["ordinal"] - 1, len(cand) - 1)], "ordinal %d before press" % q["ordinal"]
