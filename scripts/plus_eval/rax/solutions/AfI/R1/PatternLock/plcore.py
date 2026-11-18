"""Pure-python helpers for the PatternLock controller: MCP reply parsing, rotation math,
trajectory processing. No numpy needed here (vision fallbacks live in plvision.py)."""
import json
import math
import os
import time
from pathlib import Path

WORKSPACE = Path("/workspace")
OUT_DIR = Path("/workspace/output")

# ---------------------------------------------------------------- logging
_LOG_FH = None
_T0 = time.time()


def log(*parts):
    global _LOG_FH
    msg = "[%7.2fs] " % (time.time() - _T0) + " ".join(str(p) for p in parts)
    try:
        print(msg, flush=True)
    except Exception:
        pass
    try:
        if _LOG_FH is None:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            _LOG_FH = open(OUT_DIR / "controller_log.txt", "a")
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()
    except Exception:
        pass


def save_json(name, obj):
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / name, "w") as fh:
            json.dump(obj, fh, indent=1, default=str)
    except Exception as exc:  # never fatal
        log("save_json failed", name, repr(exc))


def elapsed():
    return time.time() - _T0


# ---------------------------------------------------------------- reply parsing
def reply_json(result):
    """MCP tool result -> decoded JSON metadata dict (first parseable text block)."""
    if isinstance(result, dict):
        for block in result.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "text":
                try:
                    val = json.loads(block.get("text", ""))
                    if isinstance(val, dict):
                        return val
                    return {"_value": val}
                except Exception:
                    continue
        sc = result.get("structuredContent")
        if isinstance(sc, dict):
            return sc
    return {}


def find_all(obj, key, depth=0, out=None, maxdepth=12):
    """All (depth, value) for dict entries named `key`, depth-first in order."""
    if out is None:
        out = []
    if depth > maxdepth:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append((depth, v))
            if isinstance(v, (dict, list)):
                find_all(v, key, depth + 1, out, maxdepth)
    elif isinstance(obj, list):
        for v in obj[:2000]:
            if isinstance(v, (dict, list)):
                find_all(v, key, depth + 1, out, maxdepth)
    return out


def is_num_list(v, n=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    return len(v) > 0 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                              and math.isfinite(x) for x in v)


def get_state(packet):
    """Shallowest 8-number `state` (last one at that depth)."""
    cands = [(d, v) for d, v in find_all(packet, "state") if is_num_list(v) and len(v) >= 6]
    if not cands:
        # maybe nested dict state e.g. {"values": [...]}
        for d, v in find_all(packet, "state"):
            if isinstance(v, dict):
                for vv in v.values():
                    if is_num_list(vv) and len(vv) >= 6:
                        cands.append((d + 1, vv))
    if not cands:
        return None
    dmin = min(d for d, _ in cands)
    return [float(x) for x in [v for d, v in cands if d == dmin][-1]]


def get_flag(packet, key):
    vals = [v for _, v in find_all(packet, key)]
    return any(v is True for v in vals)


def get_first(packet, key, default=None):
    vals = find_all(packet, key)
    if not vals:
        return default
    dmin = min(d for d, _ in vals)
    return [v for d, v in vals if d == dmin][0]


def resolve_path(p, base_dirs=()):
    if not p:
        return None
    pp = Path(p)
    cands = []
    if pp.is_absolute():
        cands.append(pp)
    cands.append(WORKSPACE / pp)
    for b in base_dirs:
        cands.append(Path(b) / pp)
    cands.append(Path.cwd() / pp)
    for c in cands:
        try:
            if c.exists():
                return c
        except Exception:
            pass
    return None


# ---------------------------------------------------------------- vectors / rotations
def vadd(a, b):
    return [a[i] + b[i] for i in range(3)]


def vsub(a, b):
    return [a[i] - b[i] for i in range(3)]


def vscale(a, s):
    return [a[i] * s for i in range(3)]


def vnorm(a):
    return math.sqrt(sum(x * x for x in a))


def vdot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def clip_norm(v, m):
    n = vnorm(v)
    if n <= m or n < 1e-12:
        return list(v)
    return vscale(v, m / n)


def rotvec_to_mat(r):
    th = vnorm(r)
    if th < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    k = vscale(r, 1.0 / th)
    kx, ky, kz = k
    c, s = math.cos(th), math.sin(th)
    v = 1 - c
    return [[c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s],
            [ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s],
            [kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v]]


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_T(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def mat_to_rotvec(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    c = max(-1.0, min(1.0, (tr - 1) / 2))
    th = math.acos(c)
    if th < 1e-9:
        return [0.0, 0.0, 0.0]
    if th > math.pi - 1e-4:
        # near pi: axis from the diagonal
        xx = max(0.0, (m[0][0] + 1) / 2)
        yy = max(0.0, (m[1][1] + 1) / 2)
        zz = max(0.0, (m[2][2] + 1) / 2)
        x, y, z = math.sqrt(xx), math.sqrt(yy), math.sqrt(zz)
        if x >= y and x >= z:
            y = math.copysign(y, m[0][1] + m[1][0])
            z = math.copysign(z, m[0][2] + m[2][0])
        elif y >= z:
            x = math.copysign(x, m[0][1] + m[1][0])
            z = math.copysign(z, m[1][2] + m[2][1])
        else:
            x = math.copysign(x, m[0][2] + m[2][0])
            y = math.copysign(y, m[1][2] + m[2][1])
        n = vnorm([x, y, z]) or 1.0
        return vscale([x, y, z], th / n)
    s = 2 * math.sin(th)
    axis = [(m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    return vscale(axis, th)


def rot_error(target_rv, current_rv):
    """World-frame rotation vector taking current to target: log(Rt Rc^T)."""
    return mat_to_rotvec(mat_mul(rotvec_to_mat(target_rv), mat_T(rotvec_to_mat(current_rv))))


def rot_angle_between(a_rv, b_rv):
    return vnorm(rot_error(a_rv, b_rv))


# ---------------------------------------------------------------- manifest
def load_manifest(packet):
    """Return (manifest_dict, manifest_path) from an observation packet, or (None, None)."""
    seqs = [v for _, v in find_all(packet, "frame_sequence")]
    paths = []
    for s in seqs:
        if isinstance(s, dict) and s.get("manifest_path"):
            paths.append(s["manifest_path"])
    for _, v in find_all(packet, "manifest_path"):
        if isinstance(v, str):
            paths.append(v)
    for p in paths:
        rp = resolve_path(p)
        if rp is None:
            log("manifest path not found:", p)
            continue
        try:
            return json.loads(rp.read_text()), rp
        except Exception as exc:
            log("manifest read failed", rp, repr(exc))
    return None, None


def frame_order_key(fr, idx):
    fi = fr.get("frame_index")
    if isinstance(fi, (int, float)):
        return (0, float(fi), idx)
    ts = fr.get("timestamp_s")
    if isinstance(ts, (int, float)):
        return (1, float(ts), idx)
    return (2, float(idx), idx)


def split_frames(manifest):
    frames = manifest.get("frames") if isinstance(manifest, dict) else None
    if not isinstance(frames, list):
        return [], []
    indexed = sorted([(frame_order_key(f, i), f) for i, f in enumerate(frames) if isinstance(f, dict)],
                     key=lambda t: t[0])
    ordered = [f for _, f in indexed]
    video = [f for f in ordered if f.get("phase") == "video_demo"]
    execf = [f for f in ordered if f.get("phase") != "video_demo"]
    return video, execf


def frame_file(fr, camera, kind, base_dirs=()):
    for f in fr.get("files", []) or []:
        if isinstance(f, dict) and f.get("camera") == camera and f.get("kind") == kind:
            return resolve_path(f.get("path"), base_dirs), f
    return None, None


def frame_time(fr, idx):
    ts = fr.get("timestamp_s")
    if isinstance(ts, (int, float)):
        return float(ts)
    sfi = fr.get("source_frame_index")
    if isinstance(sfi, (int, float)):
        return float(sfi) / 20.0
    return idx / 20.0


# ---------------------------------------------------------------- trajectory processing
def build_waypoints(points, rots=None, min_spacing=0.004, corner_deg=35.0):
    """points: list of [x,y,z] (time ordered). rots: list of rotvec or None.
    Returns list of dict(p, r, key, why)."""
    n = len(points)
    if n == 0:
        return []
    key = [False] * n
    why = [""] * n
    key[0] = True
    why[0] = "start"
    key[-1] = True
    why[-1] = "end"
    zs = [p[2] for p in points]
    # z local minima (presses / touches), robust over a window
    W = 3
    for k in range(n):
        lo, hi = max(0, k - W), min(n, k + W + 1)
        zwin = zs[lo:hi]
        if zs[k] <= min(zwin) + 1e-6:
            # must be clearly lower than the window's maxima on both sides (>=3 mm), over a wider window
            lo2, hi2 = max(0, k - 4 * W), min(n, k + 4 * W + 1)
            left = max(zs[lo2:k + 1]) if k > lo2 else zs[k]
            right = max(zs[k:hi2]) if hi2 > k + 1 else zs[k]
            if left - zs[k] > 0.003 and right - zs[k] > 0.003:
                key[k] = True
                why[k] = (why[k] + " zmin").strip()
    # corners: direction change between incoming/outgoing segments measured over >=1 cm arc
    for k in range(1, n - 1):
        a = None
        j = k - 1
        while j >= 0:
            if vnorm(vsub(points[k], points[j])) >= 0.01:
                a = vsub(points[k], points[j])
                break
            j -= 1
        b = None
        j = k + 1
        while j < n:
            if vnorm(vsub(points[j], points[k])) >= 0.01:
                b = vsub(points[j], points[k])
                break
            j += 1
        if a is None or b is None:
            continue
        c = vdot(a, b) / (vnorm(a) * vnorm(b) + 1e-12)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, c))))
        if ang > corner_deg:
            key[k] = True
            why[k] = (why[k] + " corner%.0f" % ang).strip()
    # dwell: robot stays within 3 mm for >=3 samples (mark the middle once)
    k = 0
    while k < n:
        j = k
        while j + 1 < n and vnorm(vsub(points[j + 1], points[k])) < 0.003:
            j += 1
        if j - k >= 2 and k > 0:
            m = (k + j) // 2
            key[m] = True
            why[m] = (why[m] + " dwell").strip()
        k = j + 1
    # non-maximum suppression of corner keys that cluster (keep the one with the largest angle/lowest z)
    wps = []
    last = None
    for k in range(n):
        p = points[k]
        if key[k] or last is None or vnorm(vsub(p, last)) >= min_spacing:
            if wps and key[k] and wps[-1]["key"] and vnorm(vsub(p, wps[-1]["p"])) < 0.004:
                # merge two near-identical keypoints: keep the lower one
                if p[2] < wps[-1]["p"][2]:
                    wps[-1] = {"p": list(p), "r": rots[k] if rots else None, "key": True,
                               "why": why[k] + "|merged", "src": k}
                continue
            wps.append({"p": list(p), "r": (list(rots[k]) if rots and rots[k] is not None else None),
                        "key": key[k], "why": why[k], "src": k})
            last = p
    return wps


def path_length(wps):
    return sum(vnorm(vsub(wps[i + 1]["p"], wps[i]["p"])) for i in range(len(wps) - 1))
