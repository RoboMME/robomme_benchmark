"""PatternLock controller: read conditioning video -> sequence of grid targets -> trace it in execution grid."""
import json, os, sys, time, io, base64
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geom import backproject
from perception import label, detect_targets, fit_grid, red_mask

WS = Path("/workspace")
OUT = WS / "output"
LOG = []

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

def blocks(reply):
    if isinstance(reply, dict):
        return reply.get("content", [])
    return reply

def packet(reply):
    for b in blocks(reply):
        if b.get("type") == "text":
            return json.loads(b["text"])
    raise RuntimeError("no json block")

def image_blocks(reply, meta):
    out = {}
    for att in meta.get("attachments", []):
        b = blocks(reply)[att["content_index"]]
        data = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(data))
        out[(att["camera"], att["kind"])] = np.array(im)
    return out

# ---------------------------------------------------------------- perception
def frame_points(fr, cam, depth):
    return backproject(depth.astype(float) * 0.001, fr["camera_intrinsics"][cam]["matrix"], fr["camera_poses"][cam])

def plausible_targets(T):
    if len(T) == 0:
        return T
    keep = (np.abs(T[:, 1]) < 0.45) & (T[:, 0] > -0.5) & (T[:, 0] < 0.3) & (T[:, 2] > 0.008) & (T[:, 2] < 0.03)
    return T[keep]

def merge_targets(lists, tol=0.025):
    pts = []
    for L in lists:
        for p in L:
            for q in pts:
                if np.linalg.norm(q[0][:2] - p[:2]) < tol:
                    q.append(p); break
            else:
                pts.append([p])
    return np.array([np.mean(q, 0) for q in pts]), [len(q) for q in pts]

def clean_grid(T, n=5):
    """Iteratively drop outliers until a clean n x n grid fit."""
    T = np.array(T)
    for _ in range(10):
        if len(T) < 4:
            break
        g = fit_grid(T, n)
        bad = (g["resid"] > 0.3) | (g["idx"].min(1) < 0) | (g["idx"].max(1) > n - 1)
        if not bad.any():
            return T, g
        # remove worst
        score = g["resid"] + 10 * ((g["idx"].min(1) < 0) | (g["idx"].max(1) > n - 1))
        T = np.delete(T, int(np.argmax(score)), 0)
    return T, fit_grid(T, n)

def grid_from_frame(fr, imgs):
    lists = []
    for cam in ("front", "wrist"):
        d = imgs.get((cam, "depth"))
        if d is None:
            continue
        P = frame_points(fr, cam, d)
        T = plausible_targets(detect_targets(P))
        lists.append(T)
        log("targets", cam, len(T))
    # wrist (top-down) is most accurate; fill gaps with bias-corrected front detections
    W = lists[1] if len(lists) > 1 else np.zeros((0, 3))
    F = lists[0] if len(lists) > 0 else np.zeros((0, 3))
    pts = [p for p in W]
    pairs = []
    for p in F:
        dd = np.linalg.norm(W[:, :2] - p[:2], axis=1) if len(W) else np.array([np.inf])
        if dd.min() < 0.025:
            pairs.append(W[int(np.argmin(dd))] - p)
    bias = np.median(np.array(pairs), 0) if len(pairs) >= 3 else np.zeros(3)
    for p in F:
        dd = np.linalg.norm(W[:, :2] - p[:2], axis=1) if len(W) else np.array([np.inf])
        if dd.min() >= 0.025:
            pts.append(p + bias)
    log("merged targets", len(pts), "front bias", np.round(bias, 4))
    best = clean_grid(np.array(pts)) if len(pts) >= 4 else None
    if best is None or len(best[0]) < 20:
        for T in lists:
            if len(T) >= 20:
                best = clean_grid(T)
                break
    T, g = best
    log("grid n=%d theta=%.1fdeg spacing=%.4f center=%s" % (len(T), np.degrees(g["theta"]), g["spacing"], np.round(g["center"], 3)))
    return T, g

def load_frame_imgs(fr):
    out = {}
    for f in fr["files"]:
        p = WS / f["path"]
        try:
            out[(f["camera"], f["kind"])] = np.array(Image.open(p))
        except Exception as e:
            log("load fail", p, e)
    return out

def video_sequence(frames, T, g):
    """Order of red activation of targets in conditioning frames."""
    s = g["spacing"]
    cnt = {}; first = {}; order = []; ee_at = {}
    for fi, fr in enumerate(frames):
        imgs = load_frame_imgs(fr)
        hits = set()
        for cam, minpx in (("front", 4), ("wrist", 12)):
            rgb = imgs.get((cam, "rgb")); d = imgs.get((cam, "depth"))
            if rgb is None or d is None:
                continue
            rgb = rgb[..., :3]
            P = frame_points(fr, cam, d)
            r = red_mask(rgb)
            for c in label(r):
                if len(c) < minpx:
                    continue
                pts = P[c[:, 0], c[:, 1]]
                pts = pts[np.isfinite(pts[:, 0])]
                if len(pts) == 0:
                    continue
                p = np.median(pts, 0)
                if not (0.0 < p[2] < 0.04):
                    continue
                dd = np.linalg.norm(T[:, :2] - p[:2], axis=1)
                k = int(np.argmin(dd))
                if dd[k] < 0.35 * s:
                    hits.add(k)
        for k in hits:
            cnt[k] = cnt.get(k, 0) + 1
            if cnt[k] == 2 and k not in order:
                order.append(k); first[k] = fi
                ee_at[k] = np.array(frames[max(0, fi - 1)]["state"][:3])
    return order, first, ee_at

# ---------------------------------------------------------------- motion
class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ended = False
        self.complete = False
        self.state = None
        self.remaining = None

    def observe(self):
        r = self.ctx.call("robomme_observe")
        m = packet(r)
        self.state = np.array(m["state"])
        lim = m.get("episode_limits", {})
        self.remaining = lim.get("remaining_episode_steps", self.remaining)
        return r, m

    def step(self, rows):
        if self.ended:
            return None
        r = self.ctx.call("robomme_step", actions=[list(map(float, x)) for x in rows])
        m = packet(r)
        if "state" in m:
            self.state = np.array(m["state"])
        lim = m.get("episode_limits", {})
        self.remaining = lim.get("remaining_episode_steps", self.remaining)
        if m.get("episode_ended"):
            self.ended = True
        tr = m.get("transition") or {}
        if tr.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True
        for st in (m.get("steps", []) or []) + (tr.get("steps", []) or []):
            if isinstance(st, dict):
                if st.get("task_complete"):
                    self.complete = True
                if st.get("episode_ended"):
                    self.ended = True
        if m.get("task_complete"):
            self.complete = True
        return m

    def move_to(self, target, vmax=0.01, tol=0.003, max_steps=150, chunk=3):
        """Closed-loop straight-line move of the EE to target xyz (world).
        The arm realises only ~40% of a commanded delta per tick, so far from the
        target we command vmax for a short chunk; near it we command the full error
        one tick at a time (geometric convergence, no overshoot seen)."""
        target = np.asarray(target, float)
        used = 0
        while used < max_steps and not self.ended:
            if self.remaining is not None and self.remaining <= 0:
                break
            err = target - self.state[:3]
            dist = np.linalg.norm(err)
            if dist < tol:
                return True
            if dist > 2.5 * vmax:
                n = chunk
                d = err / dist * vmax
            else:
                n = 1
                d = err if dist <= vmax else err / dist * vmax
            n = min(n, max_steps - used)
            if self.remaining is not None:
                n = max(1, min(n, self.remaining))
            row = list(np.clip(d / 0.05, -1, 1)) + [0, 0, 0, 1.0]
            self.step([row] * n)
            used += n
        return np.linalg.norm(target - self.state[:3]) < 3 * tol

    def hold(self, n):
        if n > 0 and not self.ended:
            self.step([[0, 0, 0, 0, 0, 0, 1.0]] * n)

# ---------------------------------------------------------------- main
def main(ctx):
    try:
        _main(ctx)
    except Exception as e:
        import traceback
        log("controller exception:", repr(e))
        log(traceback.format_exc())

def _main(ctx):
    t0 = time.time()
    rob = Robot(ctx)
    reply, meta = rob.observe()
    log("instruction:", meta.get("instruction"), "step", meta.get("step_id"), "limits", meta.get("episode_limits"))
    seq_desc = meta.get("frame_sequence")
    man = json.loads((WS / seq_desc["manifest_path"]).read_text())
    frames = sorted(man["frames"], key=lambda f: f["frame_index"])
    vid = [f for f in frames if f["phase"] == "video_demo"]
    log("video frames", len(vid))
    # video grid from first video frame
    T_v, g_v = grid_from_frame(vid[0], load_frame_imgs(vid[0]))
    order, first, ee_at = video_sequence(vid, T_v, g_v)
    seq_idx = [tuple(int(x) for x in g_v["idx"][k]) for k in order]
    log("video sequence (grid idx):", seq_idx, "frames", [first[k] for k in order])
    # tip offset & contact height from video
    offs = [ee_at[k][:2] - T_v[k][:2] for k in order]
    zs = [ee_at[k][2] for k in order]
    off = np.median(np.array(offs), 0) if offs else np.zeros(2)
    if np.linalg.norm(off) > 0.02:
        off = off / np.linalg.norm(off) * 0.02
    z_contact = float(np.median(zs)) if zs else 0.07
    z_contact = float(np.clip(z_contact, 0.06, 0.08))
    log("tip offset", np.round(off, 4), "z_contact", round(z_contact, 4))
    # execution grid
    imgs = image_blocks(reply, meta)
    T_e, g_e = grid_from_frame(meta, imgs)
    idx_e = {tuple(int(x) for x in ij): T_e[i] for i, ij in enumerate(g_e["idx"])}
    from perception import grid_to_world
    waypoints = []
    for ij in seq_idx:
        if ij in idx_e:
            p = idx_e[ij][:2]
        else:
            p = grid_to_world(g_e, ij)
            log("idx", ij, "not detected; using fitted position")
        waypoints.append(np.array([p[0] + off[0], p[1] + off[1], z_contact]))
    log("waypoints", [np.round(w, 3).tolist() for w in waypoints])
    log("perception time %.1fs" % (time.time() - t0))
    if not waypoints:
        log("no sequence found; abort")
        return
    # execute
    w0 = waypoints[0]
    rob.move_to([w0[0], w0[1], z_contact + 0.04], vmax=0.02, tol=0.004)  # v1 speed; v2 vmax=0.05 broke stick contact
    rob.move_to(w0, vmax=0.008, tol=0.002)
    rob.hold(6)
    log("at", 0, np.round(rob.state[:3], 3), "remaining", rob.remaining)
    for i, w in enumerate(waypoints[1:], 1):
        if rob.ended:
            break
        ok = rob.move_to(w, vmax=0.012, tol=0.002, max_steps=400)
        if rob.ended:
            log("episode ended during move", i, "complete", rob.complete)
            break
        rob.hold(6)
        log("at", i, np.round(rob.state[:3], 3), "remaining", rob.remaining, "ended", rob.ended, "complete", rob.complete)
    if not rob.ended:
        rob.hold(10)
    log("done ended=%s complete=%s remaining=%s time=%.1fs" % (rob.ended, rob.complete, rob.remaining, time.time() - t0))
