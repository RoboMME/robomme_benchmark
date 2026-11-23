"""PatternLock controller: read the conditioning video, recover the visited grid-node
sequence, re-detect the grid in the live scene and trace the same sequence with the stick."""
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import backproject, find_buttons

WS = Path("/workspace")
OUT = WS / "output"
LOG = []
CONTACT_Z = 0.070       # TCP height while drawing (from video contact phase)
HOVER_Z = 0.120
SNAP = 0.032
XY_OFFSET_GAIN = 1.0    # apply full lateral tip shift (ep3 evidence)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "controller_log.txt", "a") as f:
            f.write(s + "\n")
    except Exception:
        pass


def packet(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}


def reply_images(reply, p):
    """Decode rgb/depth attachments of a reply -> {(camera, kind): array}."""
    import base64, io
    out = {}
    blocks = reply.get("content", [])
    for a in p.get("attachments", []) or []:
        try:
            b = blocks[a["content_index"]]
            im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
            arr = np.array(im.convert("RGB")) if a["kind"] == "rgb" else np.array(im).astype(float) * 0.001
            out[(a["camera"], a["kind"])] = arr
        except Exception:
            pass
    return out


def node_red_pixels(reply, p, xy):
    """Count red pixels (wrist and front) whose world point is within 2.5 cm of node xy."""
    imgs = reply_images(reply, p)
    tot = 0
    for cam in ("wrist", "front"):
        if (cam, "rgb") not in imgs or (cam, "depth") not in imgs:
            continue
        pts = backproject(imgs[(cam, "depth")], p["camera_intrinsics"][cam]["matrix"], p["camera_poses"][cam])
        m = red_mask(imgs[(cam, "rgb")]) & np.isfinite(pts[..., 0])
        P = pts[m]
        if len(P):
            tot += int((np.linalg.norm(P[:, :2] - np.asarray(xy), axis=1) < 0.025).sum())
    return tot


def frame_arrays(frame, cam):
    files = {(x["camera"], x["kind"]): x["path"] for x in frame["files"]}
    rgb = np.array(Image.open(WS / files[(cam, "rgb")]).convert("RGB"))
    dep = np.array(Image.open(WS / files[(cam, "depth")])).astype(float) * 0.001
    pts = backproject(dep, frame["camera_intrinsics"][cam]["matrix"], frame["camera_poses"][cam])
    return rgb, pts


def detect_grid(frame):
    """Return 5x5x3 array of button centres indexed [row(x order), col(y order)], or None."""
    for cam in ("front", "wrist"):
        try:
            rgb, pts = frame_arrays(frame, cam)
            b = find_buttons(rgb, pts)
            c = np.array([p for p, n, _ in b if 8 <= n <= 200 and abs(p[2] - 0.016) < 0.03])
            if len(c) >= 10:
                zm = np.median(c[:, 2])
                c = c[np.abs(c[:, 2] - zm) < 0.005]
            if len(c) >= 10:
                # drop isolated candidates: keep those with >=2 neighbours at 0.06-0.13 m
                D = np.linalg.norm(c[:, None, :2] - c[None, :, :2], axis=-1)
                keep = ((D > 0.06) & (D < 0.13)).sum(1) >= 2
                c = c[keep]
            if len(c) < 20:
                log("grid", cam, "only", len(c), "candidates")
                continue
            g = fit_lattice(c)
            if g is None:
                log("grid", cam, "lattice fit failed", len(c))
                continue
            log("grid", cam, "n", len(c), "origin", np.round(g[0, 0], 3), "ex", np.round(g[1, 0] - g[0, 0], 3), "ey", np.round(g[0, 1] - g[0, 0], 3))
            return g
        except Exception as e:
            log("grid error", cam, repr(e))
    return None


def fit_lattice(c):
    """Fit a 5x5 square lattice (any small rotation) to candidate centres c (Nx3).
    Index i runs along the basis closest to world +x, j along world +y."""
    P = c[:, :2]
    D = P[None, :, :] - P[:, None, :]
    L = np.linalg.norm(D, axis=-1)
    m = (L > 0.05) & (L < 0.13)
    v = D[m]
    if len(v) < 20:
        return None
    s0 = np.median(np.linalg.norm(v, axis=1))
    near = np.abs(np.linalg.norm(v, axis=1) - s0) < 0.2 * s0
    v = v[near]
    ang = np.arctan2(v[:, 1], v[:, 0])
    th = np.angle(np.mean(np.exp(4j * ang))) / 4.0       # in (-45, 45] deg
    ex = np.array([np.cos(th), np.sin(th)]); ey = np.array([-np.sin(th), np.cos(th)])
    sp = float(np.median(np.linalg.norm(v, axis=1)))
    cen = P.mean(0)
    u = (P - cen) @ ex / sp; w = (P - cen) @ ey / sp
    iu = np.round(u - np.median(u - np.round(u))); iw = np.round(w - np.median(w - np.round(w)))
    iu -= iu.min(); iw -= iw.min()
    if iu.max() != 4 or iw.max() != 4:
        return None
    # least squares: P = o + i*a + j*b
    A = np.stack([np.ones_like(iu), iu, iw], 1)
    sol, *_ = np.linalg.lstsq(A, P, rcond=None)
    res = np.linalg.norm(A @ sol - P, axis=1)
    if np.median(res) > 0.01:
        return None
    o, a, b = sol
    z = float(np.median(c[:, 2]))
    g = np.zeros((5, 5, 3))
    for i in range(5):
        for j in range(5):
            g[i, j, :2] = o + i * a + j * b
            g[i, j, 2] = z
    return g


def stick_tip(frame):
    """World point of the visible stick end in the wrist camera (deepest near-camera pixels)."""
    files = {(x["camera"], x["kind"]): x["path"] for x in frame["files"]}
    dep = np.array(Image.open(WS / files[("wrist", "depth")])).astype(float) * 0.001
    pts = backproject(dep, frame["camera_intrinsics"]["wrist"]["matrix"], frame["camera_poses"]["wrist"])
    m = (dep > 0) & (dep < 0.25)
    if m.sum() < 50:
        return None
    dmax = dep[m].max()
    mm = m & (dep >= dmax - 0.004)
    tip = np.nanmean(pts[mm], 0)
    tcp = np.array(frame["state"][:3])
    return tip - tcp


def nearest_node(grid, p):
    d = np.linalg.norm(grid[:, :, :2] - np.asarray(p[:2]), axis=-1)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return (int(i), int(j)), float(d[i, j])


def sequence_from_states(frames, grid):
    seq = []
    zs = np.array([f["state"][2] for f in frames])
    zmin = zs.min()
    log("video z range", round(zs.min(), 3), round(zs.max(), 3))
    if zs.max() - zmin < 0.05:
        return []
    for f in frames:
        s = f["state"]
        if s[2] > zmin + 0.015:
            continue
        node, d = nearest_node(grid, s)
        if d < SNAP and (not seq or seq[-1] != node):
            seq.append(node)
    return seq, float(np.median(zs[zs < zmin + 0.015]))


def sequence_from_dwells(frames, grid, vmax=0.0018, min_len=4, near=0.022):
    """Nodes where the video stick paused at contact height (speed < vmax m/frame)."""
    S = np.array([f["state"][:3] for f in frames], float)
    if len(S) < 10 or S[:, 2].max() - S[:, 2].min() < 0.05:
        return []
    zmin = S[:, 2].min()
    ts = np.array([f.get("timestamp_s", k * 0.05) for k, f in enumerate(frames)], float)
    dt = np.maximum(np.r_[1.0, np.diff(ts)], 1e-3)
    sp = np.r_[np.inf, np.linalg.norm(np.diff(S, axis=0), axis=1)] * 0.05 / dt   # m per 20 Hz tick
    min_len = max(2, int(round(min_len * 0.05 / max(np.median(dt[1:]), 1e-3))))
    slow = (sp < vmax) & (S[:, 2] < zmin + 0.012)
    seq = []
    k = 0
    while k < len(S):
        if not slow[k]:
            k += 1
            continue
        e = k
        while e < len(S) and slow[e]:
            e += 1
        if e - k >= min_len:
            node, d = nearest_node(grid, S[k:e].mean(0))
            if d < near and (not seq or seq[-1] != node):
                seq.append(node)
        k = e
    return seq


def red_mask(rgb):
    r, g, b = [rgb[..., k].astype(float) for k in range(3)]
    return (r > 150) & (g < 90) & (b < 90)


def sequence_from_red(frames, grid):
    """Order nodes by when they turn (and stay) red in the front camera."""
    reds = []
    for f in frames[::2] + [frames[-1]]:
        cur = set()
        try:
            rgb, pts = frame_arrays(f, "front")
            m = red_mask(rgb) & np.isfinite(pts[..., 0]) & (np.abs(pts[..., 2] - grid[0, 0, 2]) < 0.01)
            P = pts[m]
            for i in range(5):
                for j in range(5):
                    if (np.linalg.norm(P[:, :2] - grid[i, j, :2], axis=1) < 0.02).sum() >= 3:
                        cur.add((i, j))
        except Exception:
            pass
        reds.append(cur)
    final = reds[-1]
    start = {}
    for n in final:
        k = len(reds) - 1
        while k > 0 and n in reds[k - 1]:
            k -= 1
        start[n] = k
    return [n for n, _ in sorted(start.items(), key=lambda t: t[1])]


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ended = False
        self.complete = False
        self.state = None
        self.step = 0
        self.remaining = 10**9
        self.last = None
        self.last_reply = None

    def observe(self):
        p = packet(self.ctx.call("robomme_observe"))
        self._upd(p)
        return p

    def _upd(self, p):
        if "state" in p:
            self.state = np.array(p["state"], float)
        if "step_id" in p:
            self.step = p["step_id"]
        lim = p.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if p.get("episode_ended"):
            self.ended = True
        tr = p.get("transition") or {}
        if tr.get("episode_ended"):
            self.ended = True
        if tr.get("task_complete"):
            self.complete = True
        for s in (p.get("steps", []) or []) + (tr.get("steps", []) or []):
            if isinstance(s, dict):
                if s.get("task_complete"):
                    self.complete = True
                if s.get("episode_ended"):
                    self.ended = True
        if p.get("task_complete"):
            self.complete = True
        self.last = p

    def send(self, rows):
        if self.ended or not rows:
            return
        rows = rows[: max(0, min(100, self.remaining))]
        if not rows:
            self.ended = True
            return
        reply = self.ctx.call("robomme_step", actions=[list(map(float, r)) for r in rows])
        p = packet(reply)
        self._upd(p)
        self.last_reply = reply

    def move_to(self, target, vmax=0.012, tol=0.003, max_steps=120, chunk=4):
        """Closed-loop straight-line move of the TCP to target (x,y,z)."""
        target = np.asarray(target, float)
        used = 0
        best, stall = 1e9, 0
        while not self.ended and used < max_steps:
            err = target - self.state[:3]
            dist = np.linalg.norm(err)
            if dist < tol:
                return True
            if dist < best - 0.002:
                best, stall = dist, 0
            else:
                stall += 1
                if stall >= 4:
                    log("move stalled at", np.round(self.state[:3], 3), "dist", round(dist, 3))
                    return False
            rows = []
            pos = self.state[:3].copy()
            for _ in range(chunk):
                e = target - pos
                dn = np.linalg.norm(e)
                if dn < tol:
                    break
                stepv = e if dn <= vmax else e / dn * vmax
                pos = pos + stepv
                rows.append(list(np.clip(stepv / 0.05, -1, 1)) + [0, 0, 0, 0])
            if not rows:
                return True
            self.send(rows)
            used += len(rows)
        return np.linalg.norm(target - self.state[:3]) < 3 * tol


DWELL_MIN, DWELL_MAX = 12, 24
TRAVEL_V = 0.015
HARD_STEPS = 440        # observed native end at step 450 regardless of reported limit


def dwell(rob, xy, cz, dmin=None, dmax=None):
    """Hold the stick on node xy until it turns red (or DWELL_MAX steps)."""
    held = 0
    red = 0
    dmin = DWELL_MIN if dmin is None else dmin
    dmax = DWELL_MAX if dmax is None else dmax
    while not rob.ended and held < dmax:
        n = 4
        rows = []
        pos = rob.state[:3]
        e = np.array([xy[0], xy[1], cz]) - pos
        rows.append(list(np.clip(e / 0.05, -0.2, 0.2)) + [0, 0, 0, 0])
        rows += [[0, 0, 0, 0, 0, 0, 0]] * (n - 1)
        rob.send(rows)
        held += n
        try:
            red = node_red_pixels(rob.last_reply, rob.last, xy)
        except Exception as ex:
            red = -1
        if held >= dmin and red >= 60:
            break
    return held, red


def main(ctx):
    t0 = time.time()
    log("PatternLock controller v5 start (dwell + full tip offset + fast)")
    rob = Robot(ctx)
    p = rob.observe()
    seqd = p.get("frame_sequence") or {}
    frames = []
    try:
        man = json.loads((WS / seqd["manifest_path"]).read_text())
        frames = sorted(man["frames"], key=lambda f: f["frame_index"])
    except Exception as e:
        log("manifest error", repr(e))
    video = [f for f in frames if f.get("phase") == "video_demo"]
    exe = [f for f in frames if f.get("phase") == "execution"]
    log("frames", len(frames), "video", len(video), "exec", len(exe))

    live_grid = detect_grid(exe[-1]) if exe else None
    vid_grid = detect_grid(video[0]) if video else None
    if live_grid is None:
        live_grid = vid_grid
    if vid_grid is None:
        vid_grid = live_grid
    if live_grid is None:
        log("no grid found; abort")
        return

    seq, cz = [], CONTACT_Z
    try:
        r = sequence_from_states(video, vid_grid)
        if r:
            seq, cz = r
    except Exception as e:
        log("state seq error", repr(e), traceback.format_exc())
    log("state(snap) sequence", seq, "contact z", round(cz, 4))
    try:
        seq_dw = sequence_from_dwells(video, vid_grid)
    except Exception as e:
        seq_dw = []
        log("dwell seq error", repr(e))
    log("dwell sequence", seq_dw)
    if len(seq_dw) >= 2:
        seq = seq_dw
    try:
        seq_red = sequence_from_red(video, vid_grid)
        log("red sequence", seq_red)
    except Exception as e:
        seq_red = []
        log("red seq error", repr(e))
    if len(seq) < 2 and len(seq_red) >= 2:
        seq = seq_red
    if len(seq) < 2:
        log("no usable sequence; abort")
        return
    cz = float(np.clip(cz, 0.060, 0.080))
    off = np.zeros(3)
    try:
        tv = stick_tip(video[0]) if video else None
        tl = stick_tip(exe[-1]) if exe else None
        log("tip rel tcp video", None if tv is None else np.round(tv, 4), "live", None if tl is None else np.round(tl, 4))
        if tv is not None and tl is not None:
            same_pose = np.linalg.norm(np.array(video[0]["state"][:3]) - np.array(exe[-1]["state"][:3])) < 0.01
            off = tl - tv
            if not same_pose:
                log("warning: video/live start poses differ")
            off = np.clip(off, [-0.03, -0.03, -0.03], [0.03, 0.03, 0.09])
            off[:2] *= XY_OFFSET_GAIN
    except Exception as e:
        log("tip error", repr(e))
    log("tool offset (live - video)", np.round(off, 4))
    cz = cz - off[2]
    targets = [live_grid[i, j, :2] - off[:2] for i, j in seq]
    log("targets", [np.round(t, 3).tolist() for t in targets], "t", round(time.time() - t0, 1))

    path = sum(np.linalg.norm(np.asarray(targets[k + 1]) - targets[k]) for k in range(len(targets) - 1))
    travel = int(path / TRAVEL_V * 1.2) + 70
    spare = max(0, min(rob.remaining, HARD_STEPS) - travel - 10)
    dmax = int(np.clip(spare / len(targets), 4, DWELL_MAX))
    dmin = min(DWELL_MIN, dmax)
    log("path", round(path, 3), "travel est", travel, "remaining", rob.remaining, "dwell", dmin, dmax)
    # approach above first node, descend, trace, stay
    x0, y0 = targets[0]
    rob.move_to([x0, y0, HOVER_Z - off[2]], vmax=0.03, tol=0.004, max_steps=60)
    log("hover", np.round(rob.state[:3], 3), "step", rob.step)
    rob.move_to([x0, y0, cz], vmax=0.008, tol=0.002, max_steps=30)
    log("down", np.round(rob.state[:3], 3), "step", rob.step)
    h, red = dwell(rob, targets[0], cz, dmin, dmax)
    log("dwell node 0", seq[0], "held", h, "red", red, "step", rob.step, "ended", rob.ended, "complete", rob.complete)
    for k, (x, y) in enumerate(targets[1:], 1):
        if rob.ended:
            break
        ok = rob.move_to([x, y, cz], vmax=TRAVEL_V, tol=0.003, max_steps=80)
        log("node", k, seq[k], ok, np.round(rob.state[:3], 3), "step", rob.step, "ended", rob.ended, "complete", rob.complete)
        if not rob.ended:
            h, red = dwell(rob, (x, y), cz, dmin, dmax)
            log("dwell node", k, seq[k], "held", h, "red", red, "step", rob.step, "ended", rob.ended, "complete", rob.complete)
    # dwell a few steps at the last node so contact registers
    if not rob.ended:
        x, y = targets[-1]
        rob.move_to([x, y, cz], vmax=0.004, tol=0.001, max_steps=6, chunk=6)
    log("done step", rob.step, "ended", rob.ended, "complete", rob.complete, "t", round(time.time() - t0, 1))
