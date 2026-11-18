"""Robot I/O and motion primitives over ctx.call (robomme_* tools)."""
import base64, io, json, math, time
import numpy as np
from PIL import Image

POS_SCALE = 0.05   # m per unit action
ROT_SCALE = 0.5    # rad per unit action
DOWN_RV = np.array([-math.pi, 0.0, 0.0])  # reset orientation: gripper pointing down


class EpisodeEnded(Exception):
    pass


def log(*a):
    print("[%.1f]" % time.time(), *a, flush=True)


def parse(result):
    """Return (meta, images) where images = {(cam, kind): ndarray}."""
    content = result.get("content", [])
    meta = {}
    for b in content:
        if b.get("type") == "text":
            try:
                meta = json.loads(b["text"]); break
            except Exception:
                pass
    imgs = {}
    for att in meta.get("attachments", []) or []:
        i = att.get("content_index")
        if i is None or i >= len(content):
            continue
        b = content[i]
        if b.get("type") != "image":
            continue
        try:
            im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
            arr = np.array(im)
            if att["kind"] == "depth":
                arr = arr.astype(float) * 0.001
            else:
                arr = np.array(im.convert("RGB"))
            imgs[(att["camera"], att["kind"])] = arr
        except Exception as e:
            log("image decode failed", att, e)
    return meta, imgs


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None
        self.imgs = {}
        self.ended = False
        self.step_id = 0
        self.remaining = None
        self.history = []   # list of (step_id, sequence manifest path)

    def _absorb(self, result):
        meta, imgs = parse(result)
        if meta:
            self.meta = meta
            if imgs:
                self.imgs = imgs
            self.step_id = meta.get("step_id", self.step_id)
            lim = meta.get("episode_limits") or {}
            self.remaining = lim.get("remaining_episode_steps", self.remaining)
            if meta.get("episode_ended") or (meta.get("transition") or {}).get("episode_ended"):
                self.ended = True
            fs = meta.get("frame_sequence") or {}
            if fs.get("manifest_path"):
                self.history.append((self.step_id, fs["manifest_path"]))
        return meta

    def observe(self):
        return self._absorb(self.ctx.call("robomme_observe"))

    @property
    def state(self):
        return np.array(self.meta["state"], float)

    @property
    def pos(self):
        return self.state[:3]

    @property
    def rv(self):
        return self.state[3:6]

    def step(self, rows):
        if self.ended:
            raise EpisodeEnded()
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        if self.remaining is not None:
            rows = rows[:max(1, self.remaining)]
        meta = self._absorb(self.ctx.call("robomme_step", actions=rows))
        if meta.get("task_complete"):
            log("task_complete reported", meta.get("task_complete"))
        if self.ended:
            raise EpisodeEnded()
        return meta

    def cam(self, cam):
        m = self.meta
        return (self.imgs.get((cam, "rgb")), self.imgs.get((cam, "depth")),
                m["camera_intrinsics"][cam]["matrix"], m["camera_poses"][cam])


# ---------- rotation helpers ----------
def rv_to_R(rv):
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def R_to_rv(R):
    c = (np.trace(R) - 1) / 2
    th = math.acos(max(-1, min(1, c)))
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from diagonal
        k = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        i = int(np.argmax(k))
        k[(i + 1) % 3] *= np.sign(R[i, (i + 1) % 3]) or 1
        k[(i + 2) % 3] *= np.sign(R[i, (i + 2) % 3]) or 1
        return k / np.linalg.norm(k) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def target_R(yaw):
    """Down-pointing gripper rotated by yaw about world z."""
    return Rz(yaw) @ rv_to_R(DOWN_RV)


def rot_error(R_cur, R_tgt):
    """World-frame rotation vector taking current to target."""
    return R_to_rv(R_tgt @ R_cur.T)


def yaw_of(rv):
    """Yaw about world z of current gripper relative to the reset orientation."""
    R = rv_to_R(rv) @ rv_to_R(DOWN_RV).T
    return math.atan2(R[1, 0], R[0, 0])


# ---------- motion ----------
ALPHA = 0.42   # measured: per row the TCP moves ~0.4 of the commanded delta (ep1, steps 188-629)


def move_to(rb, target, yaw=None, grip=-1.0, vmax=0.018, wmax=0.15, tol=0.004, ytol=0.03,
            max_iter=10, verbose=True, max_rows=50, callback=None):
    """Closed-loop move of the TCP to target (world m), optional yaw, gripper command held.
    vmax: achieved m per step (<= ~0.02 because one action unit is 0.05 m and ALPHA~0.4)."""
    target = np.asarray(target, float)
    Rt = target_R(yaw) if yaw is not None else None
    err = 1e9
    for it in range(max_iter):
        p = rb.pos
        d = target - p
        err = float(np.linalg.norm(d))
        rerr = np.zeros(3)
        if Rt is not None:
            rerr = rot_error(rv_to_R(rb.rv), Rt)
        rn = float(np.linalg.norm(rerr))
        if err < tol and rn < ytol:
            break
        n = max(1, int(math.ceil(max(err / vmax, rn / wmax))))
        n = min(n, max_rows)
        dp = d / (n * ALPHA) / POS_SCALE
        dr = rerr / (n * ALPHA) / ROT_SCALE
        sc = max(1.0, float(np.abs(dp).max()), float(np.abs(dr).max()))
        row = list(dp / sc) + list(dr / sc) + [grip]
        s0 = rb.step_id
        rb.step([row] * n)
        if callback is not None:
            callback()
        if verbose:
            log("move it%d err=%.4f rerr=%.3f n=%d steps=%d -> pos %s" % (it, err, rn, n, rb.step_id - s0, rb.pos.round(4)))
    return float(np.linalg.norm(target - rb.pos))


def hold(rb, n, grip, callback=None, chunk=4):
    k = 0
    while k < n:
        m = min(chunk, n - k)
        rb.step([[0, 0, 0, 0, 0, 0, grip]] * m)
        k += m
        if callback is not None:
            callback()
