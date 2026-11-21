"""Defensive MCP reply parsing and logging.

The exact reply schema is not known offline, so every accessor searches the
decoded JSON recursively and tolerates missing fields.
"""
import json
import os
import sys
import time
import traceback

OUT_DIR = "/workspace/output"
_LOG_FH = None
_T0 = time.time()


def log(*parts):
    global _LOG_FH
    msg = "[%7.1fs] " % (time.time() - _T0) + " ".join(str(p) for p in parts)
    try:
        print(msg, flush=True)
    except Exception:
        pass
    try:
        if _LOG_FH is None:
            os.makedirs(OUT_DIR, exist_ok=True)
            _LOG_FH = open(os.path.join(OUT_DIR, "controller_log.txt"), "a")
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()
    except Exception:
        pass


def log_exc(where):
    log("EXC in", where, traceback.format_exc()[-2000:])


_SAVED = {}


def save_json(name, obj, limit_bytes=400000, max_count=3):
    """Save a JSON object under /workspace/output (a few per name)."""
    try:
        n = _SAVED.get(name, 0)
        if n >= max_count:
            return
        _SAVED[name] = n + 1
        s = json.dumps(obj, default=str)
        if len(s) > limit_bytes:
            s = s[:limit_bytes]
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "%s_%d.json" % (name, n)), "w") as fh:
            fh.write(s)
    except Exception:
        pass


def reply_json(result):
    """First text block decoded as JSON (dict); {} if impossible."""
    try:
        for b in result.get("content", []):
            if b.get("type") == "text":
                t = b.get("text", "")
                try:
                    v = json.loads(t)
                    if isinstance(v, dict):
                        return v
                    return {"_value": v}
                except Exception:
                    continue
    except Exception:
        pass
    sc = result.get("structuredContent") if isinstance(result, dict) else None
    if isinstance(sc, dict):
        return sc
    return {}


def strip_images(result):
    """Copy of a result without base64 image data (for logging)."""
    try:
        out = []
        for b in result.get("content", []):
            if b.get("type") == "text":
                out.append(b)
            else:
                out.append({k: (v if k != "data" else "<%d bytes>" % len(str(v))) for k, v in b.items()})
        return {"content": out}
    except Exception:
        return {}


def find_key(obj, names, depth=0, maxdepth=8):
    """First value whose key is in names (depth-first, dicts before lists)."""
    if depth > maxdepth:
        return None
    if isinstance(obj, dict):
        for k in names:
            if k in obj:
                return obj[k]
        for v in obj.values():
            if isinstance(v, (dict, list)):
                r = find_key(v, names, depth + 1, maxdepth)
                if r is not None:
                    return r
    elif isinstance(obj, list):
        for v in obj[:50]:
            if isinstance(v, (dict, list)):
                r = find_key(v, names, depth + 1, maxdepth)
                if r is not None:
                    return r
    return None


def find_all_keys(obj, name, depth=0, maxdepth=8, acc=None):
    if acc is None:
        acc = []
    if depth > maxdepth:
        return acc
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == name:
                acc.append(v)
            if isinstance(v, (dict, list)):
                find_all_keys(v, name, depth + 1, maxdepth, acc)
    elif isinstance(obj, list):
        for v in obj[:600]:
            if isinstance(v, (dict, list)):
                find_all_keys(v, name, depth + 1, maxdepth, acc)
    return acc


def _is_num_list(v, n):
    return isinstance(v, list) and len(v) == n and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def get_state(j):
    """8-D state from a reply json, or None."""
    for cand in find_all_keys(j, "state"):
        if _is_num_list(cand, 8):
            return [float(x) for x in cand]
        if isinstance(cand, dict):
            for k in ("values", "vector", "data"):
                if _is_num_list(cand.get(k), 8):
                    return [float(x) for x in cand[k]]
    for k in ("proprio", "robot_state", "proprioception"):
        for cand in find_all_keys(j, k):
            if _is_num_list(cand, 8):
                return [float(x) for x in cand]
    return None


def truthy(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes", "success", "succeeded")
    return False


def episode_ended(j):
    for k in ("episode_ended",):
        for v in find_all_keys(j, k):
            if truthy(v):
                return True
    return False


def task_complete(j):
    for k in ("task_complete",):
        for v in find_all_keys(j, k):
            if truthy(v):
                return True
    return False


def get_step_id(j):
    v = j.get("step_id") if isinstance(j, dict) else None
    if isinstance(v, (int, float)):
        return int(v)
    for v in find_all_keys(j, "step_id"):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    return None


def steps_remaining(j):
    """Best-effort remaining-step count from episode_limits; None if unknown."""
    lim = find_key(j, ["episode_limits"])
    if isinstance(lim, dict):
        for k in ("steps_remaining", "remaining_steps", "remaining"):
            v = lim.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        mx = None
        for k in ("max_steps", "step_limit", "max_episode_steps", "limit", "horizon"):
            v = lim.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                mx = int(v)
                break
        used = None
        for k in ("steps_used", "elapsed_steps", "step", "steps", "step_id", "current_step"):
            v = lim.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                used = int(v)
                break
        if mx is not None and used is not None:
            return mx - used
        if mx is not None:
            sid = get_step_id(j)
            if sid is not None:
                return mx - sid
    return None


def step_limit(j):
    lim = find_key(j, ["episode_limits"])
    if isinstance(lim, dict):
        for k in ("max_steps", "step_limit", "max_episode_steps", "limit", "horizon"):
            v = lim.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
    for k in ("max_steps", "step_limit", "max_episode_steps"):
        v = find_key(j, [k])
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    return None


def manifest_paths(j):
    out = []
    for v in find_all_keys(j, "manifest_path"):
        if isinstance(v, str):
            out.append(v)
    return out


def get_instruction(j):
    v = find_key(j, ["instruction", "goal", "language_instruction"])
    return v if isinstance(v, str) else ""
