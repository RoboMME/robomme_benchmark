"""Tolerant MCP reply parsing for the RoboMME-Plus robot tools.

The exact reply layout was never observed during development (R1: the only
simulator start is the rehearsal), so every accessor searches the reply
structure for the documented field names instead of assuming one path.
"""
import base64
import io
import json
import os
import time

import numpy as np
from PIL import Image

WORKSPACE = "/workspace"


def find_all(obj, key, path=(), out=None, depth=0):
    """All (path, value) pairs whose dict key equals `key`, depth-first."""
    if out is None:
        out = []
    if depth > 12:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append((path + (k,), v))
            if isinstance(v, (dict, list)):
                find_all(v, key, path + (k,), out, depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:2000]):
            if isinstance(v, (dict, list)):
                find_all(v, key, path + (i,), out, depth + 1)
    return out


def is_numlist(v, n=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    return len(v) > 0 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def summarize(obj, maxlen=400, depth=0):
    """Compact structural summary for logs (no image bytes)."""
    if depth > 5:
        return "..."
    if isinstance(obj, dict):
        return {k: summarize(v, maxlen, depth + 1) for k, v in list(obj.items())[:40]}
    if isinstance(obj, list):
        if is_numlist(obj) and len(obj) <= 16:
            return obj
        if len(obj) > 6:
            return [summarize(v, maxlen, depth + 1) for v in obj[:3]] + [f"...({len(obj)} items)"]
        return [summarize(v, maxlen, depth + 1) for v in obj]
    if isinstance(obj, str) and len(obj) > maxlen:
        return obj[:maxlen] + "..."
    return obj


class Reply:
    def __init__(self, raw):
        self.raw = raw if isinstance(raw, dict) else {}
        self.meta = {}
        self.texts = []
        self.images = []
        last_text = ""
        for b in self.raw.get("content", []) or []:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                txt = b.get("text", "") or ""
                self.texts.append(txt)
                parsed = None
                try:
                    parsed = json.loads(txt)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict) and not self.meta:
                    self.meta = parsed
                elif isinstance(parsed, dict):
                    self.meta.setdefault("_extra", []).append(parsed)
                else:
                    last_text = txt
            elif t == "image":
                im = None
                try:
                    im = Image.open(io.BytesIO(base64.b64decode(b.get("data", ""))))
                    im.load()
                except Exception:
                    im = None
                info = {k: v for k, v in b.items() if k != "data"}
                self.images.append({"img": im, "info": info, "pre": last_text[:200]})
                last_text = ""
        sc = self.raw.get("structuredContent")
        if isinstance(sc, dict):
            if not self.meta:
                self.meta = sc
            else:
                self.meta.setdefault("_structured", sc)

    # ---- generic accessors -------------------------------------------------
    def first(self, key, pred=None, prefer_short=True, avoid=("steps", "frames", "_extra")):
        cands = find_all(self.meta, key)
        if pred is not None:
            cands = [c for c in cands if pred(c[1])]
        if not cands:
            return None
        good = [c for c in cands if not any(a in c[0] for a in avoid)]
        pool = good or cands
        if prefer_short:
            pool = sorted(pool, key=lambda c: len(c[0]))
            return pool[0][1]
        return pool[-1][1]

    def state(self):
        s = self.first("state", pred=lambda v: is_numlist(v, 8))
        if s is None:
            # last element of a step list
            cands = [c for c in find_all(self.meta, "state") if is_numlist(c[1], 8)]
            if cands:
                s = cands[-1][1]
        return None if s is None else np.array(s, dtype=float)

    def flag(self, key):
        vals = [v for _, v in find_all(self.meta, key) if isinstance(v, bool)]
        return any(vals)

    def ended(self):
        return self.flag("episode_ended")

    def complete(self):
        return self.flag("task_complete")

    def step_id(self):
        v = self.first("step_id", pred=lambda v: isinstance(v, int) and not isinstance(v, bool))
        return v

    def instruction(self):
        v = self.first("instruction", pred=lambda v: isinstance(v, str) and len(v) > 3)
        if v is None:
            v = self.first("goal", pred=lambda v: isinstance(v, str) and len(v) > 3)
        return v

    def limits(self):
        return self.first("episode_limits")


def remaining_steps(limits):
    """Best-effort remaining step count from an episode_limits value."""
    if not isinstance(limits, dict):
        return None
    for k, v in limits.items():
        if "remain" in k.lower() and isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    mx = None
    used = None
    for k, v in limits.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        kl = k.lower()
        if ("max" in kl or "limit" in kl or "horizon" in kl) and "step" in kl:
            mx = int(v)
        if ("used" in kl or "elapsed" in kl or kl in ("step", "steps", "step_id", "current_step")):
            used = int(v)
    if mx is not None and used is not None:
        return mx - used
    return None


def max_steps(limits):
    if not isinstance(limits, dict):
        return None
    for k, v in limits.items():
        kl = k.lower()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and \
                ("max" in kl or "limit" in kl or "horizon" in kl) and "step" in kl:
            return int(v)
    return None


class Logger:
    def __init__(self, path="/workspace/output/controller_log.txt"):
        self.t0 = time.time()
        self.path = path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.f = open(path, "a", buffering=1)
        except Exception:
            self.f = None

    def __call__(self, *args):
        msg = "[%7.1fs] " % (time.time() - self.t0) + " ".join(str(a) for a in args)
        try:
            print(msg, flush=True)
        except Exception:
            pass
        if self.f:
            try:
                self.f.write(msg + "\n")
            except Exception:
                pass

    def elapsed(self):
        return time.time() - self.t0


def load_image(path):
    p = path if os.path.isabs(path) else os.path.join(WORKSPACE, path)
    im = Image.open(p)
    im.load()
    return im
