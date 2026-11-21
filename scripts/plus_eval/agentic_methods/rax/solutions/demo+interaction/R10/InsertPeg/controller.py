"""InsertPeg controller: replay the current episode's conditioning-video EE trajectory
closed-loop (the execution scene resets to the video's layout), with tight convergence
at the grasp and at the end of insertion."""
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlib import packet, make_row, rot_err, quat_to_R
from holevis import feats, hole3d
import absvis as AV
from PIL import Image

WS = Path("/workspace")
OUT = WS / "output"
LOG = None

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    try:
        with open(OUT / "controller_log.txt", "a") as f:
            f.write(s + "\n")
    except Exception:
        pass

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx; self.ended = False; self.success = None
        self.state = None; self.remaining = None; self.step_id = 0
    def observe(self):
        p = packet(self.ctx.call("robomme_observe"))
        self._upd(p); return p
    def _upd(self, p):
        if "state" in p: self.state = np.array(p["state"], float)
        lim = p.get("episode_limits") or {}
        if "remaining_episode_steps" in lim: self.remaining = lim["remaining_episode_steps"]
        self.step_id = p.get("step_id", self.step_id)
        if p.get("episode_ended"): self.ended = True
        for s in (p.get("transition") or {}).get("steps", []) or []:
            if s.get("task_complete") is not None:
                self.success = s.get("task_complete")
            if s.get("episode_ended"): self.ended = True
    def step(self, rows):
        if self.ended: return None
        if self.remaining is not None and self.remaining <= 0: self.ended = True; return None
        if self.remaining is not None: rows = rows[: max(1, self.remaining)]
        p = packet(self.ctx.call("robomme_step", actions=rows))
        self._upd(p); return p

def err(cur, tgt):
    return float(np.linalg.norm(cur[:3] - tgt[:3])), float(np.linalg.norm(rot_err(cur[3:6], tgt[3:6])))

def converge(rb, tgt, grip, ptol, rtol, maxn, gain=1.0):
    n = 0
    while n < maxn and not rb.ended:
        pe, re_ = err(rb.state, tgt)
        if pe < ptol and re_ < rtol: break
        rb.step([make_row(rb.state, tgt, grip, pgain=gain, rgain=gain)]); n += 1
    return n

def load_video(p0):
    seq = p0.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    m = None
    if mp and (WS / mp).exists():
        m = json.loads((WS / mp).read_text())
        if not any(f.get("phase") == "video_demo" for f in m.get("frames", [])):
            m = None
    if m is None:
        eid = p0.get("episode_id")
        cands = sorted((WS / "runtime/autonomous_controller/runs").glob("*/frames/sequence_000000/manifest.json"),
                       key=lambda q: q.stat().st_mtime, reverse=True)
        for c in cands:
            mm = json.loads(c.read_text())
            if (mm.get("transition") or {}).get("episode_id") == eid:
                m = mm; log("video manifest via fallback", c); break
    fr = [f for f in m["frames"] if f.get("phase") == "video_demo"]
    fr.sort(key=lambda f: f.get("frame_index", 0))
    S = np.array([f["state"] for f in fr], float)
    load_video.frames = fr
    return S

def find_insert_start(S, path_len=0.065):
    """index after which the remaining EE path to the final pose is <= path_len"""
    P = S[:, :3]
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    rem = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    # last static tail
    idx = int(np.nonzero(rem > path_len)[0][-1]) if np.any(rem > path_len) else 0
    return idx

def insert_phase(rb, Sentry, Sfinal, grip, t_end):
    """v4: approach to contact, then slide the peg tip over the face along an Archimedean
    spiral under light pressure; when the depth jumps (peg drops into the hole) push to the end."""
    p0 = Sentry[:3].copy(); pf = Sfinal[:3].copy()
    L = float(np.linalg.norm(pf - p0)); n = (pf - p0) / max(L, 1e-9)
    a = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    u1 = np.cross(n, a); u1 /= np.linalg.norm(u1); u2 = np.cross(n, u1)
    rot = Sfinal[3:6]
    log("insert v4: L", round(L, 4), "n", np.round(n, 3).tolist())
    depth = lambda: float(np.dot(rb.state[:3] - p0, n))
    lat = lambda: (float(np.dot(rb.state[:3] - p0, u1)), float(np.dot(rb.state[:3] - p0, u2)))

    def push_through(o1, o2, maxsteps=60):
        s = depth(); stall = 0; last = s; k = 0
        while not rb.ended and stall < 4 and k < maxsteps:
            tgt = np.concatenate([p0 + o1 * u1 + o2 * u2 + n * min(L, s + 0.004), rot, [0, 0]])
            rb.step([make_row(rb.state, tgt, grip, pmax=0.12)]); k += 1
            s = depth()
            if s > L - 0.002: break
            stall = stall + 1 if s - last < 0.0006 else 0
            last = s
            if rb.remaining is not None and rb.remaining < 3: break
        return s

    def finish(o1, o2):
        tgt = np.concatenate([pf + o1 * u1 + o2 * u2, rot, [0, 0]])
        for _ in range(8):
            if rb.ended: break
            rb.step([make_row(rb.state, tgt, grip)])

    # straight approach to first contact
    converge(rb, np.concatenate([p0, rot, [0, 0]]), grip, 0.0012, 0.008, 25)
    s = push_through(0.0, 0.0, 80)
    log("contact try depth", round(s, 4), "/", round(L, 4), "rem", rb.remaining)
    if s > L - 0.004 or rb.ended:
        finish(0.0, 0.0); return True
    s_c = s
    # dense hexagonal grid of discrete pushes around the estimate (hole accepts only ~0.5 mm error)
    g = 0.0007; pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * g + j * g / 2; y = j * g * np.sqrt(3) / 2
            rr = np.hypot(x, y)
            if 1e-9 < rr <= 0.00165: pts.append((rr, np.arctan2(y, x), x, y))
    pts.sort()
    for ti, (rr, _, o1, o2) in enumerate(pts):
        if rb.ended or time.time() > t_end: break
        if rb.remaining is not None and rb.remaining < 150: break
        converge(rb, np.concatenate([p0 + o1 * u1 + o2 * u2 + n * (s_c - 0.0035), rot, [0, 0]]), grip, 0.0015, 0.03, 3)
        converge(rb, np.concatenate([p0 + o1 * u1 + o2 * u2 + n * (s_c - 0.0035), rot, [0, 0]]), grip, 0.0006, 0.01, 6)
        s = push_through(o1, o2, 40)
        log("grid try", ti, "off", np.round([o1 * 1000, o2 * 1000], 2).tolist(), "depth", round(s, 4), "rem", rb.remaining)
        if s > L - 0.004 or rb.ended:
            finish(o1, o2); return True
        if s > s_c + 0.004:
            # partially entered: keep pushing with tiny lateral nudges around the measured position
            l1, l2 = lat()
            for w1, w2 in [(0, 0), (0.0002, 0), (-0.0002, 0), (0, 0.0002), (0, -0.0002), (0, 0)]:
                if rb.ended or (rb.remaining is not None and rb.remaining < 40): break
                s = push_through(l1 + w1, l2 + w2, 15)
                log("  partial nudge", w1, w2, "depth", round(s, 4), "rem", rb.remaining)
                if s > L - 0.004 or rb.ended:
                    finish(l1 + w1, l2 + w2); return True
    converge(rb, np.concatenate([p0 + n * (s_c - 0.004), rot, [0, 0]]), grip, 0.001, 0.02, 6)
    s_c = push_through(0.0, 0.0, 30)
    if s_c > L - 0.004 or rb.ended:
        finish(0.0, 0.0); return True
    rounds = [(0.0006, 0.0045, 0.0005), (0.0007, 0.0075, 0.0005), (0.0005, 0.0045, 0.0004)]
    thr = 0.0018; nfalse = 0
    for ri, (pitch, rmax, ds) in enumerate(rounds):
        th = 0.0 if ri != 2 else np.pi / 2; r = 0.0
        while r < rmax and not rb.ended and time.time() < t_end:
            if rb.remaining is not None and rb.remaining < 12: return False
            r = pitch * th / (2 * np.pi)
            th += ds / max(r, 0.0008)
            o1, o2 = r * np.cos(th), r * np.sin(th)
            tgt = np.concatenate([p0 + o1 * u1 + o2 * u2 + n * (s_c + 0.003), rot, [0, 0]])
            rb.step([make_row(rb.state, tgt, grip, pmax=0.12)])
            s = depth()
            if s > s_c + thr:
                l1, l2 = lat()
                log("drop detected r mm", round(r * 1000, 2), "lat mm", round(l1 * 1000, 2), round(l2 * 1000, 2), "depth", round(s, 4), "rem", rb.remaining)
                s2 = push_through(l1, l2, 80)
                log("  pushed to", round(s2, 4), "/", round(L, 4), "rem", rb.remaining, "ended", rb.ended)
                if s2 > L - 0.004 or rb.ended:
                    finish(l1, l2); return True
                nfalse += 1
                if nfalse % 3 == 0: thr = min(thr + 0.0005, 0.004)
                # false drop: back out to the contact depth and continue the spiral
                converge(rb, np.concatenate([p0 + o1 * u1 + o2 * u2 + n * (s_c - 0.003), rot, [0, 0]]), grip, 0.001, 0.02, 6)
        log("spiral round", ri, "exhausted, rem", rb.remaining)
        # back off and re-establish the contact depth
        converge(rb, np.concatenate([p0 + n * (s_c - 0.006), rot, [0, 0]]), grip, 0.001, 0.02, 8)
        s_c = push_through(0.0, 0.0, 30)
        if s_c > L - 0.004 or rb.ended:
            finish(0.0, 0.0); return True
    return False

def wrist_depth_from_frame(f):
    files = {(x["camera"], x["kind"]): x["path"] for x in f["files"]}
    return np.array(Image.open(WS / files[("wrist", "depth")]))

def basis(n):
    a = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    u1 = np.cross(n, a); u1 /= np.linalg.norm(u1); u2 = np.cross(n, u1)
    if u2[2] < 0: u2 = -u2
    return u1, u2

def hole_edges(F, frame, n):
    """lateral centre (along u1) and vertical edges (along u2) of the hole opening on the face plane."""
    if not F or not F.get("hole") or F["hole"]["n"] < 60: return None
    h = F["hole"]; u0, uu1, v0, v1 = h["bbox"]
    _, P = hole3d(F, frame, quat_to_R)
    e1, e2 = basis(n)
    A = P @ e1; B = P @ e2
    out = {"a": None, "bmin": None, "bmax": None}
    if u0 > 2 and uu1 < 253:
        out["a"] = float((np.percentile(A, 2) + np.percentile(A, 98)) / 2)
    if v0 > 2: out["bmax"] = float(np.percentile(B, 98))
    if v1 < F["tip_v"] - 3: out["bmin"] = float(np.percentile(B, 2))
    return out

def hole_ok(F):
    if not F or not F.get("hole"): return False
    h = F["hole"]; u0, u1, v0, v1 = h["bbox"]
    return h["n"] >= 80 and u0 > 2 and u1 < 253 and v1 < F["tip_v"] - 3

def choose_ref(frames, S, i_ins, n, p0):
    best = None
    for i in range(max(0, i_ins - 30), len(S)):
        s = float(np.dot(S[i, :3] - p0, n))
        if s > 0.016: break
        try:
            F = feats(wrist_depth_from_frame(frames[i]))
            E = hole_edges(F, frames[i], n)
        except Exception:
            continue
        if not E or E["a"] is None or E["bmin"] is None or E["bmax"] is None: continue
        score = F["hole"]["n"] - 5000 * abs(s - 0.004)
        if best is None or score > best[0]:
            F["E"] = E
            best = (score, i, F)
    return best

def live_feats(rb, p=None):
    p = p or rb.observe()
    seq = p.get("frame_sequence") or {}
    # depth image of the current observation from the content blocks is not decoded here;
    # read the published frame file instead
    mp = seq.get("manifest_path")
    if mp and (WS / mp).exists():
        m = json.loads((WS / mp).read_text()); f = m["frames"][-1]
        return feats(wrist_depth_from_frame(f)), f
    return None, None

def measure_live(rb, tgt, grip, n):
    converge(rb, tgt, grip, 0.0008, 0.006, 25)
    p = packet(rb.ctx.call("robomme_step", actions=[make_row(rb.state, tgt, grip)]))
    rb._upd(p)
    F, f = live_feats(rb, p)
    return (hole_edges(F, f, n) if F else None), F

def servo_hole(rb, Sref, Fref, n, grip, iters=3):
    """lateral EE offset (perp to n) from live vs video hole edges (world, on the face plane)"""
    e1, e2 = basis(n)
    R = Fref["E"]
    off = np.zeros(3); found = False
    for it in range(iters):
        da = db = None
        for back in ([0.0, -0.008, 0.006, -0.016] if it == 0 else [0.0, -0.008, 0.006]):
            tgt = Sref.copy(); tgt[:3] = Sref[:3] + off + n * back
            E, F = measure_live(rb, tgt, grip, n)
            log("  measure it", it, "back", back, "bbox", None if not F else (F.get("hole") or {}).get("bbox"), "tip", None if not F else F.get("tip_v"), "E", E)
            if rb.ended: return off, found
            if E and E["a"] is not None:
                da = E["a"] - R["a"]
                dbs = []
                if E["bmax"] is not None: dbs.append(E["bmax"] - R["bmax"])
                if E["bmin"] is not None: dbs.append(E["bmin"] - R["bmin"])
                if dbs:
                    db = float(np.mean(dbs)); break
        if da is None or db is None:
            log("servo: hole not measurable", da, db)
            if da is not None and not found:
                off = da * e1; found = True
            return off, found
        newoff = da * e1 + db * e2
        if np.linalg.norm(newoff) > 0.02:
            newoff = newoff / np.linalg.norm(newoff) * 0.02
        log("servo it", it, "da mm", round(da * 1000, 2), "db mm", round(db * 1000, 2), "offset mm", np.round(newoff * 1000, 2).tolist())
        found = True
        conv = np.linalg.norm(newoff - off) < 0.0007
        off = newoff
        if conv: break
    return off, found

ABS_BIAS = (-0.0012, 0.00275)   # (e1, e2) bias of the absolute estimate, calibrated on 2 conditioning videos

def abs_align(rb, Sbase, S_ins, n, grip):
    """sweep along n around Sbase; on frames with the full hole visible, estimate the insertion
    line from the peg axis vs hole centre; return its perpendicular offset from S_ins (or None)."""
    e1, e2 = AV.basis(n)
    pts = []
    first = True
    for back in np.arange(-0.012, 0.0161, 0.0015):
        if rb.ended: break
        if rb.remaining is not None and rb.remaining < 260: break
        tgt = Sbase.copy(); tgt[:3] = Sbase[:3] + n * back
        converge(rb, tgt, grip, 0.0008, 0.006, 15 if first else 4); first = False
        p = packet(rb.ctx.call("robomme_step", actions=[make_row(rb.state, tgt, grip)])); rb._upd(p)
        mp = (p.get("frame_sequence") or {}).get("manifest_path")
        if not mp or not (WS / mp).exists(): continue
        f = json.loads((WS / mp).read_text())["frames"][-1]
        try:
            M = AV.measure(wrist_depth_from_frame(f), f["camera_poses"]["wrist"], n)
        except Exception as e:
            log("abs measure error", repr(e)); continue
        if not M or not M.get("hole"): 
            log("  abs back", back, "no hole"); continue
        h = M["hole"]; da, db = AV.required_shift(M)
        st = np.array(f["state"][:3], float)
        line = st + (da - ABS_BIAS[0]) * e1 + (db - ABS_BIAS[1]) * e2
        rel = line - S_ins[:3]; rel = rel - n * float(rel @ n)
        log("  abs back", round(float(back), 4), "full", h["full"], "bbox", h["bbox"], "w/h mm", round(h["w"] * 1e3, 1), round(h["h"] * 1e3, 1),
            "peg w", round(M["peg_w"] * 1e3, 1), "off mm", np.round(rel * 1e3, 2).tolist())
        if h["full"] and h["n"] > 300 and h["h"] > 0.0115: pts.append(rel)
    if len(pts) < 2: return None, len(pts)
    P = np.array(pts); med = np.median(P, axis=0)
    return med, len(pts)

def _main(ctx):
    try:
        OUT.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    t0 = time.time(); t_end = t0 + 1000
    rb = Robot(ctx)
    p0 = rb.observe()
    S = load_video(p0)
    log("video frames", len(S), "steps remaining", rb.remaining)
    fsum = S[:, 6] + S[:, 7]
    open_w = float(np.max(fsum))
    events = []
    for i in range(1, len(S)):
        if fsum[i] < fsum[i-1] - 0.002 and fsum[i-1] > open_w - 0.01 and (not events or events[-1][1] != "close"):
            events.append((i, "close"))
    log("gripper events", events)
    close_idx = [i for i, e in events if e == "close"]
    i_ins = find_insert_start(S)
    log("insert start index", i_ins, np.round(S[i_ins], 4).tolist())
    grip = -1.0
    i = 0
    lag_max = 3
    while i <= i_ins and not rb.ended:
        if close_idx and i == close_idx[0]:
            tgt = S[i - 1]
            n = converge(rb, tgt, -1.0, 0.002, 0.02, 40)
            log("pre-grasp converge", n, "err", err(rb.state, tgt), "state", np.round(rb.state, 4).tolist())
            grip = 1.0
            for _ in range(12):
                rb.step([make_row(rb.state, tgt, grip)])
            log("after close fingers", rb.state[6:8].tolist())
            close_idx.pop(0)
            while i < len(S) - 1 and fsum[i] < fsum[i-1] - 1e-4: i += 1
            continue
        tgt = S[i]
        grip = 1.0 if (fsum[i] < open_w - 0.01) else -1.0
        rb.step([make_row(rb.state, tgt, grip)])
        k = 0
        while k < lag_max and not rb.ended:
            pe, re_ = err(rb.state, tgt)
            if pe < 0.01 and re_ < 0.08: break
            rb.step([make_row(rb.state, tgt, grip)]); k += 1
        if i % 20 == 0:
            log("i", i, "err", np.round(err(rb.state, tgt), 4).tolist(), "rem", rb.remaining)
        i += 1
    if not rb.ended:
        log("fingers before insert", rb.state[6:8].tolist())
        off = np.zeros(3)
        try:
            p0 = S[i_ins, :3]; pf = S[-1, :3]; n = (pf - p0) / np.linalg.norm(pf - p0)
            ref = choose_ref(load_video.frames, S, i_ins, n, p0)
            if ref is not None:
                log("ref frame", ref[1], "hole", ref[2]["hole"], "peg", ref[2]["peg_u"], ref[2]["tip_v"])
                off, ok = servo_hole(rb, S[ref[1]].copy(), ref[2], n, grip)
                try:
                    Sb = S[ref[1]].copy(); Sb[:3] += off
                    offa, na = abs_align(rb, Sb, S[i_ins], n, grip)
                    log("abs offset", None if offa is None else np.round(offa * 1000, 2).tolist(), "from", na, "frames; servo off", np.round(off * 1000, 2).tolist())
                    if offa is not None and np.linalg.norm(offa - off) < 0.012:
                        off = offa
                except Exception as e:
                    log("abs error", repr(e), traceback.format_exc())
            else:
                log("no ref frame with visible hole")
        except Exception as e:
            log("servo error", repr(e), traceback.format_exc())
            off = np.zeros(3)
        log("lateral offset mm", np.round(off * 1000, 2).tolist())
        Sent = S[i_ins].copy(); Sent[:3] += off
        Sfin = S[-1].copy(); Sfin[:3] += off
        ok = insert_phase(rb, Sent, Sfin, grip, t_end)
        log("insert result", ok)
    # idle hold until end only if not ended (do nothing more)
    log("done ended", rb.ended, "success", rb.success, "remaining", rb.remaining, "t", round(time.time() - t0, 1))


def main(ctx):
    try:
        _main(ctx)
    except Exception as e:
        log("FATAL controller error", repr(e), traceback.format_exc())
