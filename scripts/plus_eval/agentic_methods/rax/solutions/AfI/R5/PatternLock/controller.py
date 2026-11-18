"""PatternLock controller: read the conditioning video's stick trajectory (per-frame state
and red-lit buttons as fallback), snap it to the 5x5 button grid, then retrace the same
node sequence on the execution grid with the drawing stick."""
import json, os, sys, time, traceback
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geom import backproject  # noqa

WS = Path("/workspace")
OUT = WS / "output"
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def meta_of(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}


def load_png(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im)


# ---------------- grid detection ----------------
def cluster1d(vals, k, min_gap):
    vals = np.sort(np.asarray(vals))
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] > min_gap:
            groups.append([v])
        else:
            groups[-1].append(v)
    return [(float(np.mean(g)), len(g)) for g in groups]


def best5(clusters):
    """Choose 5 consecutive clusters with the most regular spacing (and most points)."""
    cl = sorted([c for c in clusters if c[1] >= 2])
    best, bscore = [c[0] for c in cl], None
    for a in range(0, len(cl) - 4):
        sub = cl[a:a + 5]
        pos = np.array([c[0] for c in sub])
        gaps = np.diff(pos)
        if gaps.min() < 0.04 or gaps.max() > 0.16:
            continue
        score = np.std(gaps) / np.mean(gaps) - 0.01 * sum(c[1] for c in sub)
        if bscore is None or score < bscore:
            best, bscore = list(pos), score
    return best


def detect_grid(rgb, dep_m, K, pose):
    """Return (xs[5], ys[5]) world coords of grid rows/cols or None."""
    from scipy import ndimage
    rgb = rgb.astype(int)[..., :3]
    white = (rgb.min(axis=2) > 160) & (np.abs(rgb[..., 0] - rgb[..., 2]) < 45)
    red = (rgb[..., 0] > 150) & (rgb[..., 1] < 60) & (rgb[..., 2] < 60)
    mask = white | red
    lab, n = ndimage.label(mask)
    pts = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < 3:
            continue
        d = dep_m[ys, xs]
        d = d[d > 0]
        if len(d) == 0:
            continue
        p = backproject(xs.mean(), ys.mean(), float(np.median(d)), K, pose)
        if 0.0 < p[2] < 0.035:
            pts.append(p)
    if len(pts) < 10:
        return None
    pts = np.array(pts)
    xs = best5(cluster1d(pts[:, 0], 5, 0.03))
    ys = best5(cluster1d(pts[:, 1], 5, 0.03))
    if len(xs) != 5 or len(ys) != 5:
        log("grid clusters", len(xs), len(ys), xs, ys)
        return None
    return np.array(xs), np.array(ys)


def regular_grid(g):
    """Fit a regular grid (center + spacing) to cluster coords for robustness."""
    xs, ys = g
    idx = np.arange(5) - 2
    ax = np.polyfit(idx, xs, 1)
    ay = np.polyfit(idx, ys, 1)
    return np.polyval(ax, idx), np.polyval(ay, idx)


# ---------------- pattern extraction ----------------
def snap_sequence(states, grid, tol=0.025, zmargin=0.012):
    xs, ys = grid
    st = np.array([s[:3] for s in states])
    zmin = st[:, 2].min()
    seq = []
    for p in st:
        if p[2] > zmin + zmargin:
            continue
        i = int(np.argmin(np.abs(xs - p[0])))
        j = int(np.argmin(np.abs(ys - p[1])))
        if abs(xs[i] - p[0]) < tol and abs(ys[j] - p[1]) < tol:
            if not seq or seq[-1] != (i, j):
                seq.append((i, j))
    return seq, float(zmin)


def red_sequence(frames, grid, K, pose):
    """Fallback: order buttons by first frame they appear red in the front view."""
    from geom import project
    xs, ys = grid
    first = {}
    for k, fr in enumerate(frames):
        files = {(f["camera"], f["kind"]): f["path"] for f in fr["files"]}
        if ("front", "rgb") not in files:
            continue
        rgb = load_png(WS / files[("front", "rgb")]).astype(int)
        for i in range(5):
            for j in range(5):
                if (i, j) in first:
                    continue
                u, v, _ = project([xs[i], ys[j], 0.016], K, pose)
                u, v = int(round(u)), int(round(v))
                patch = rgb[max(v - 2, 0):v + 3, max(u - 3, 0):u + 4]
                r = (patch[..., 0] > 150) & (patch[..., 1] < 60) & (patch[..., 2] < 60)
                if r.sum() >= 3:
                    first[(i, j)] = k
    return [c for c, _ in sorted(first.items(), key=lambda t: t[1])]


# ---------------- motion ----------------
def front_red(reply, m):
    import base64, io
    from PIL import Image
    for a in m.get("attachments", []):
        if a.get("camera") == "front" and a.get("kind") == "rgb":
            blk = reply["content"][a["content_index"]]
            im = np.asarray(Image.open(io.BytesIO(base64.b64decode(blk["data"]))).convert("RGB")).astype(int)
            return int(((im[..., 0] > 150) & (im[..., 1] < 60) & (im[..., 2] < 60)).sum())
    return -1


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None
        self.ended = False
        self.complete = False

    def observe(self):
        self.meta = meta_of(self.ctx.call("robomme_observe"))
        return self.meta

    def state(self):
        return np.array(self.meta["state"])

    def remaining(self):
        return self.meta.get("episode_limits", {}).get("remaining_episode_steps", 0)

    def step(self, rows):
        r = self.ctx.call("robomme_step", actions=[list(map(float, x)) for x in rows])
        m = meta_of(r)
        try:
            self.red = front_red(r, m)
            if self.red != getattr(self, "_last_red", 0):
                log("red change", getattr(self, "_last_red", 0), "->", self.red, "step", m.get("step_id"),
                    "pos", np.round(m.get("state", [0, 0, 0])[:3], 4).tolist())
                self._last_red = self.red
        except Exception as e:
            log("red err", e)
        self.meta = m
        tr = m.get("transition") or {}
        if tr.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True
        for s in (m.get("steps", []) or []) + (tr.get("steps", []) or []):
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        if m.get("task_complete"):
            self.complete = True
        if m.get("episode_ended"):
            self.ended = True
        return m

    def move_to(self, target, vmax=0.012, tol=0.004, max_steps=80, **_):
        """Closed-loop straight move; vmax is the achieved speed per step (m)."""
        target = np.asarray(target, float)
        used = 0
        while used < max_steps and not self.ended and self.remaining() > 2:
            err = target - self.state()[:3]
            if np.linalg.norm(err) < tol:
                return True
            d = err.copy()
            nd = np.linalg.norm(d)
            if nd > vmax:
                d = d / nd * vmax
            d = d * 2.2
            self.step([[d[0] / 0.05, d[1] / 0.05, d[2] / 0.05, 0, 0, 0, 0]])
            used += 1
        return np.linalg.norm(target - self.state()[:3]) < tol * 2


def find_initial_sequence(rob, m0):
    fs = m0.get("frame_sequence") or {}
    mp = fs.get("manifest_path")
    if not mp:
        return None
    man = json.loads((WS / mp).read_text())
    return man


def main(ctx):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rob = Robot(ctx)
    try:
        run(rob)
    except Exception:
        log("EXC", traceback.format_exc())
    finally:
        log("done complete=%s ended=%s t=%.1f" % (rob.complete, rob.ended, time.time() - t0))
        try:
            (OUT / "controller_log.txt").write_text("\n".join(LOG))
        except Exception:
            pass


def rotvec_err(target_rv, cur_rv):
    from scipy.spatial.transform import Rotation as Rot
    e = Rot.from_rotvec(target_rv) * Rot.from_rotvec(cur_rv).inv()
    return e.as_rotvec()


def red_count(meta_reply_paths=None):
    return None


GAIN = 2.2  # measured: one step achieves ~0.40x of the commanded delta


def replay(rob, traj, offset, max_rows=1):
    """Track the per-step trajectory with measured-pose feedback (one row per call)."""
    n = len(traj)
    k = 1
    while k < n and not rob.ended:
        if rob.remaining() <= 3:
            log("replay: step budget low")
            break
        cur = rob.state()
        tgt = np.asarray(traj[k][:3]) + offset
        d = (tgt - cur[:3]) * GAIN
        nd = np.linalg.norm(d)
        if nd > 0.045:
            d = d / nd * 0.045
        rv = np.clip(rotvec_err(np.asarray(traj[k][3:6]), cur[3:6]), -0.05, 0.05)
        row = list(np.clip(np.concatenate([d / 0.05, rv / 0.5, [0.0]]), -1, 1))
        rob.step([row])
        k += 1
    return k


def resample_by_source(vframes):
    """Per-control-step trajectory from (possibly subsampled) video frames using source_frame_index."""
    pairs = [(f.get("source_frame_index"), f["state"]) for f in vframes if f.get("state")]
    if not pairs or any(p[0] is None for p in pairs):
        return [np.asarray(f["state"], float) for f in vframes if f.get("state")]
    pairs.sort(key=lambda q: q[0])
    idx = np.array([p[0] for p in pairs], float)
    st = np.array([p[1] for p in pairs], float)
    out_idx = np.arange(int(idx[0]), int(idx[-1]) + 1)
    out = np.zeros((len(out_idx), st.shape[1]))
    for c in range(3):
        out[:, c] = np.interp(out_idx, idx, st[:, c])
    near = np.searchsorted(idx, out_idx).clip(0, len(idx) - 1)
    out[:, 3:] = st[near, 3:]
    return [r for r in out]


def stick_bias(states, L):
    """Median offset between video stick dwell points and the depth-fitted button centroids."""
    st = np.array([q[:3] for q in states])
    if len(st) < 5 or L is None:
        return np.zeros(2)
    sp = np.median(np.linalg.norm(np.diff(L, axis=1), axis=2))
    flat = L.reshape(-1, 2)
    v = np.linalg.norm(np.diff(st[:, :2], axis=0), axis=1)
    zmin = st[:, 2].min()
    errs = []
    for k in range(1, len(st)):
        if v[k - 1] < 0.0006 and st[k, 2] < zmin + 0.01:
            d = st[k, :2] - flat
            i = int(np.linalg.norm(d, axis=1).argmin())
            if np.linalg.norm(d[i]) < 0.2 * sp:
                errs.append(d[i])
    if len(errs) < 3:
        return np.zeros(2)
    return np.clip(np.median(np.array(errs), axis=0), -0.012, 0.012)


def snap_nodes(states, L, zmargin=0.012):
    st = np.array([q[:3] for q in states])
    zmin = float(st[:, 2].min())
    sp = np.median(np.linalg.norm(np.diff(L, axis=1), axis=2))
    flat = L.reshape(-1, 2)
    seq = []
    for p in st:
        if p[2] > zmin + zmargin:
            continue
        d = np.linalg.norm(flat - p[:2], axis=1)
        k = int(d.argmin())
        if d[k] < 0.25 * sp:
            ij = (k // 5, k % 5)
            if not seq or seq[-1] != ij:
                seq.append(ij)
    return seq, zmin


def red_sequence_nodes(frames, L, K, pose):
    from geom import project
    first = {}
    for k, fr in enumerate(frames):
        files = {(f["camera"], f["kind"]): f["path"] for f in fr["files"]}
        if ("front", "rgb") not in files:
            continue
        rgb = load_png(WS / files[("front", "rgb")]).astype(int)
        for i in range(5):
            for j in range(5):
                if (i, j) in first:
                    continue
                u, v, _ = project([L[i, j, 0], L[i, j, 1], 0.016], K, pose)
                u, v = int(round(u)), int(round(v))
                patch = rgb[max(v - 2, 0):v + 3, max(u - 3, 0):u + 4]
                r = (patch[..., 0] > 150) & (patch[..., 1] < 60) & (patch[..., 2] < 60)
                if r.sum() >= 3:
                    first[(i, j)] = k
    return [c for c, _ in sorted(first.items(), key=lambda q: q[1])]


def synth_traj(cur, pts, speed=0.0045, dwell=10):
    rv = np.asarray(cur[3:6], float)
    out = [np.concatenate([cur[:3], rv])]

    def line(a, b):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / speed)))
        for t in range(1, n + 1):
            out.append(np.concatenate([a + (b - a) * t / n, rv]))

    p0 = pts[0]
    pre = p0 + np.array([-0.014, 0.0, 0.002])
    line(np.asarray(cur[:3], float), pre)
    line(pre, p0)
    for _ in range(dwell):
        out.append(out[-1].copy())
    for a, b in zip(pts[:-1], pts[1:]):
        line(a, b)
        for _ in range(dwell):
            out.append(out[-1].copy())
    return out


def run(rob):
    m0 = rob.observe()
    log("instruction:", m0.get("instruction"), "step", m0.get("step_id"), "limits", m0.get("episode_limits"))
    man = find_initial_sequence(rob, m0)
    frames = man["frames"] if man else []
    vframes = [f for f in frames if f.get("phase") == "video_demo"]
    log("frames", len(frames), "video", len(vframes))

    ex = [f for f in frames if f.get("phase") == "execution"]
    exf = ex[-1] if ex else None
    from lattice import lattice_from_frame, affine_between

    def lat_of(fr):
        try:
            files = {(f["camera"], f["kind"]): f["path"] for f in fr["files"]}
            rgb = load_png(WS / files[("front", "rgb")])
            dep = load_png(WS / files[("front", "depth")]).astype(float) * 0.001
            return lattice_from_frame(rgb, dep, fr["camera_intrinsics"]["front"]["matrix"], fr["camera_poses"]["front"])
        except Exception as e:
            log("lattice err", e)
            return None

    Lex = lat_of(exf) if exf else None
    Lvid = None
    for fr in vframes[:3]:
        Lvid = lat_of(fr)
        if Lvid is not None:
            break
    corners = lambda L: None if L is None else np.round(L[[0, 0, 4, 4], [0, 4, 0, 4]], 3).tolist()
    log("exec lattice corners", corners(Lex))
    log("video lattice corners", corners(Lvid))
    A, t = np.eye(2), np.zeros(2)
    states0 = [f["state"] for f in vframes if f.get("state")]
    bias = stick_bias(states0, Lvid) if states0 else np.zeros(2)
    log("stick bias", np.round(bias, 4).tolist())
    if Lex is not None and Lvid is not None:
        A, t = affine_between(Lvid + bias, Lex + bias)
        sv = np.linalg.svd(A)[1]
        if sv.max() > 1.5 or sv.min() < 0.6 or np.linalg.norm(t) > 0.4:
            log("affine implausible; using identity", A.tolist(), t.tolist())
            A, t = np.eye(2), np.zeros(2)
    else:
        log("lattice missing; identity mapping")
    log("affine A", np.round(A, 4).tolist(), "t", np.round(t, 4).tolist())
    if Lex is None:
        Lex = Lvid
    if Lvid is None and Lex is not None:
        Lvid = (Lex.reshape(-1, 2) - t) @ np.linalg.inv(A).T
        Lvid = Lvid.reshape(5, 5, 2)

    states = [f["state"] for f in vframes if f.get("state")]
    seq = []
    zdraw = 0.07
    if states and Lvid is not None:
        seq, zdraw = snap_nodes(states, Lvid)
        log("video node seq", seq, "zmin", round(zdraw, 4))
    traj = resample_by_source(vframes) if states else []
    log("trajectory steps", len(traj), "from video frames", len(states))
    zs = np.array([q[2] for q in traj]) if traj else np.array([1.0])
    if len(traj) < 20 or zs.min() > 0.12:
        log("no usable video states; synthesizing trajectory from red-lit node order")
        if Lvid is None or not vframes:
            log("cannot recover pattern")
            return
        fr = vframes[0]
        seq = red_sequence_nodes(vframes, Lvid, fr["camera_intrinsics"]["front"]["matrix"], fr["camera_poses"]["front"])
        log("red seq", seq)
        if len(seq) < 1:
            return
        if not (0.03 < zdraw < 0.12):
            zdraw = 0.07
        traj = synth_traj(rob.state(), [np.array([Lex[i, j, 0], Lex[i, j, 1], zdraw]) for i, j in seq])
    else:
        mapped = []
        for q in traj:
            q = q.copy()
            q[:2] = A @ q[:2] + t
            mapped.append(q)
        traj = mapped
    p_now = rob.state()[:3]
    log("start pose", np.round(p_now, 3).tolist(), "traj start", np.round(traj[0][:3], 3).tolist(), "len", len(traj))
    if np.linalg.norm(p_now - traj[0][:3]) > 0.01:
        rob.move_to(traj[0][:3], vmax=0.015, tol=0.005, max_steps=60)
    offset = np.zeros(3)
    k = replay(rob, traj, offset)
    log("replayed to frame", k, "state", np.round(rob.state()[:3], 4).tolist(), "step", rob.meta.get("step_id"),
        "complete", rob.complete, "ended", rob.ended)
    hold = 0
    while not rob.ended and hold < 20 and rob.remaining() > 2:
        tgt = traj[-1][:3]
        d = np.clip((tgt - rob.state()[:3]) / 0.05, -0.2, 0.2)
        rob.step([list(d) + [0, 0, 0, 0]] * 2)
        hold += 2
    log("final state", np.round(rob.state()[:3], 4).tolist(), "step", rob.meta.get("step_id"),
        "complete", rob.complete, "ended", rob.ended)
