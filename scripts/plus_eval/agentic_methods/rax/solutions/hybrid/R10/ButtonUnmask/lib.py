"""Helpers for ButtonUnmask controller: decode MCP replies, geometry, motion."""
import base64, io, json, math, os, time
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")
LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

def parse(reply):
    blocks = reply.get("content", [])
    meta = None
    for b in blocks:
        if b.get("type") == "text":
            try:
                meta = json.loads(b["text"]); break
            except Exception:
                pass
    out = {"meta": meta or {}, "rgb": {}, "depth": {}}
    if meta is None:
        return out
    for att in meta.get("attachments", []):
        idx = att.get("content_index")
        if idx is None or idx >= len(blocks):
            continue
        b = blocks[idx]
        data = b.get("data")
        if not data:
            continue
        im = Image.open(io.BytesIO(base64.b64decode(data)))
        if att.get("kind") == "rgb":
            out["rgb"][att["camera"]] = np.array(im.convert("RGB"))
        elif att.get("kind") == "depth":
            out["depth"][att["camera"]] = np.array(im).astype(np.float32) * 0.001
    return out

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def cam_params(meta, cam):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], float)
    pose = meta["camera_poses"][cam]
    return K, quat_to_R(pose["quaternion_wxyz"]), np.array(pose["position_m"], float)

def deproject(meta, cam, u, v, d):
    K, R, t = cam_params(meta, cam)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.array([(u - cx) * d / fx, -(v - cy) * d / fy, -d])
    return R @ pc + t

def project(meta, cam, p):
    K, R, t = cam_params(meta, cam)
    pc = R.T @ (np.asarray(p, float) - t)
    d = -pc[2]
    u = pc[0] * K[0, 0] / d + K[0, 2]
    v = -pc[1] * K[1, 1] / d + K[1, 2]
    return u, v, d

def depth_at(depth, u, v, r=1):
    h, w = depth.shape
    u0, v0 = int(round(u)), int(round(v))
    win = depth[max(0, v0-r):v0+r+1, max(0, u0-r):u0+r+1]
    win = win[win > 0]
    return float(np.median(win)) if win.size else None

def load_seq_frames_all(meta):
    """Return list of dicts {step, rgb:{cam}, depth:{cam}, meta} for saved intermediate frames."""
    seq = meta.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    res = []
    if not mp:
        return res
    try:
        man = json.loads((WS / mp).read_text())
    except Exception as e:
        log("seq manifest read fail", e)
        return res
    for fr in man.get("frames", []):
        item = {"step": fr.get("step_id"), "rgb": {}, "depth": {}, "meta": fr}
        for f in fr.get("files", []):
            try:
                im = Image.open(WS / f["path"])
                if f["kind"] == "rgb":
                    item["rgb"][f["camera"]] = np.array(im.convert("RGB"))
                else:
                    item["depth"][f["camera"]] = np.array(im).astype(np.float32) * 0.001
            except Exception:
                pass
        res.append(item)
    return res

def load_seq_frames(meta):
    """Return list of (step_id, front_rgb, meta_frame) for saved intermediate frames."""
    seq = meta.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    res = []
    if not mp:
        return res
    try:
        man = json.loads((WS / mp).read_text())
    except Exception as e:
        log("seq manifest read fail", e)
        return res
    for fr in man.get("frames", []):
        files = {(f["camera"], f["kind"]): f["path"] for f in fr.get("files", [])}
        p = files.get(("front", "rgb"))
        if not p:
            continue
        try:
            rgb = np.array(Image.open(WS / p).convert("RGB"))
        except Exception:
            continue
        dpath = files.get(("front", "depth"))
        dep = None
        if dpath:
            try:
                dep = np.array(Image.open(WS / dpath)).astype(np.float32) * 0.001
            except Exception:
                dep = None
        res.append((fr.get("step_id"), rgb, dep, fr))
    return res

class Robot:
    def __init__(self, ctx, outdir="/workspace/output"):
        self.ctx = ctx
        self.out = Path(outdir); self.out.mkdir(parents=True, exist_ok=True)
        self.last = None
        self.ended = False
        self.complete = False
        self.frames = []  # (step, front_rgb, front_depth, framemeta)
        self.n_calls = 0
        self.hold_yaw = 0.0
        self.on_frames = None

    @property
    def meta(self):
        return self.last["meta"]

    @property
    def state(self):
        return np.array(self.meta["state"], float)

    @property
    def step_id(self):
        return int(self.meta.get("step_id", 0))

    @property
    def remaining(self):
        lim = self.meta.get("episode_limits") or {}
        return int(lim.get("remaining_episode_steps", 0))

    def observe(self):
        r = parse(self.ctx.call("robomme_observe"))
        self.last = r
        self._check_end(r["meta"])
        return r

    def _check_end(self, meta):
        if meta.get("episode_ended"):
            self.ended = True
        for s in meta.get("steps", []) or []:
            if isinstance(s, dict):
                if s.get("task_complete"):
                    self.complete = True
                if s.get("episode_ended"):
                    self.ended = True
        if meta.get("task_complete"):
            self.complete = True
        tr = meta.get("transition") or {}
        if tr.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True
        for s in tr.get("steps", []) or []:
            if isinstance(s, dict):
                if s.get("task_complete"):
                    self.complete = True
                if s.get("episode_ended"):
                    self.ended = True

    def step(self, rows, keep_frames=False):
        if self.ended:
            return self.last
        rows = [[float(max(-1, min(1, x))) for x in r] for r in rows]
        r = parse(self.ctx.call("robomme_step", actions=rows))
        self.n_calls += 1
        if not r["meta"].get("state"):
            # keep old state if missing
            r["meta"]["state"] = self.meta.get("state") if self.last else None
        self.last = r
        self._check_end(r["meta"])
        if keep_frames:
            try:
                fr = load_seq_frames_all(r["meta"])
                self.frames.extend(fr)
                if self.on_frames:
                    self.on_frames(fr)
            except Exception as e:
                log("frame load error", repr(e))
        return r

    def move_to(self, target, grip=-1.0, tol=0.004, max_steps=60, chunk=3, vmax=1.0,
                keep_frames=False, yaw=None, rtol=0.03, rmax=0.5):
        """Closed-loop Cartesian move (+ optional yaw servo). Returns (pos error, steps used)."""
        target = np.asarray(target, float)
        used = 0
        while used < max_steps and not self.ended:
            st = self.state
            pos = st[:3]
            rerr = rot_err(st[3:6], yaw if yaw is not None else self.hold_yaw)
            if np.linalg.norm(target - pos) < tol and np.linalg.norm(rerr) < rtol:
                break
            rows = []
            p = pos.copy(); r = rerr.copy()
            n = min(chunk, max_steps - used)
            for _ in range(n):
                d = np.clip((target - p) / 0.05, -vmax, vmax)
                dr = np.clip(r / 0.5, -rmax, rmax)
                rows.append(list(d) + list(dr) + [grip])
                p = p + d * 0.05; r = r - dr * 0.5
                if np.linalg.norm(target - p) < tol * 0.5 and np.linalg.norm(r) < rtol * 0.5:
                    break
            self.step(rows, keep_frames=keep_frames)
            used += len(rows)
        if yaw is not None:
            self.hold_yaw = yaw
        return float(np.linalg.norm(target - self.state[:3])), used

    def rot_correction(self, target_rv=(-math.pi, 0, 0), gain=0.5):
        """Small correction to keep gripper pointing down with initial yaw."""
        return [0.0, 0.0, 0.0]

    def hold(self, n, grip, keep_frames=False):
        rows = [[0, 0, 0, 0, 0, 0, grip]] * n
        return self.step(rows, keep_frames=keep_frames)

    def save_img(self, name, cam="front"):
        try:
            Image.fromarray(self.last["rgb"][cam]).save(self.out / f"{name}_{cam}.png")
        except Exception as e:
            log("save fail", e)

def rotvec_to_R(rv):
    rv = np.asarray(rv, float); th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c)); th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if abs(th - math.pi) < 1e-3:
        # near pi
        M = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(M), 0))
        i = int(np.argmax(k)); k = M[:, i] / math.sqrt(max(M[i, i], 1e-9))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th

R_DOWN = rotvec_to_R([-math.pi, 0, 0])

def target_R(yaw):
    cz, sz = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ R_DOWN

def rot_err(rv_now, yaw):
    """World-frame rotvec that takes current orientation to target."""
    Rn = rotvec_to_R(rv_now)
    return R_to_rotvec(target_R(yaw) @ Rn.T)

def cur_yaw(rv_now):
    Rn = rotvec_to_R(rv_now)
    M = Rn @ R_DOWN.T
    return math.atan2(M[1, 0], M[0, 0])
