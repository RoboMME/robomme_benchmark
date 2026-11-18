"""Motion + observation helpers for RoboMME MoveCube (RAX ctx API)."""
import base64, io, json, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from PIL import Image

LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)

def _meta(result):
    for b in result.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}

def decode_images(result, meta):
    """Return dict {(camera, kind): np.array} from MCP image blocks."""
    out = {}
    blocks = result.get("content", [])
    for att in meta.get("attachments", []):
        i = att.get("content_index")
        if i is None or i >= len(blocks):
            continue
        b = blocks[i]
        data = b.get("data")
        if not data:
            continue
        img = Image.open(io.BytesIO(base64.b64decode(data)))
        if att["kind"] == "depth":
            arr = np.array(img).astype(np.float32) * 0.001
        else:
            arr = np.array(img.convert("RGB"))
        out[(att["camera"], att["kind"])] = arr
    return out

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.step_id = 0
        self.ended = False
        self.complete = False
        self.remaining = None
        self.last_meta = None
        self.ik_errors = 0
        self.penalty_flag = False

    def observe(self):
        r = self.ctx.call("robomme_observe")
        m = _meta(r)
        self.last_meta = m
        self._upd(m)
        imgs = decode_images(r, m)
        return m, imgs

    def _upd(self, m):
        if "state" in m and m["state"]:
            self.state = np.array(m["state"], dtype=float)
        if "step_id" in m:
            self.step_id = m["step_id"]
        lim = m.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if m.get("episode_ended"):
            self.ended = True
        for s in m.get("steps", []) or []:
            if isinstance(s, dict):
                if s.get("task_complete"):
                    self.complete = True
                if s.get("episode_ended"):
                    self.ended = True
        if m.get("task_complete"):
            self.complete = True
        tr = m.get("transition") or {}
        if isinstance(tr, dict):
            if tr.get("task_complete"):
                self.complete = True
            if tr.get("episode_ended"):
                self.ended = True

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        if self.remaining is not None:
            rows = rows[:max(0, self.remaining)]
        if not rows:
            return None
        before = self.step_id
        r = self.ctx.call("robomme_step", actions=rows)
        m = _meta(r)
        self.last_meta = m
        self._upd(m)
        adv = self.step_id - before
        if adv > len(rows) + 2:
            self.ik_errors += 1
            self.penalty_flag = True
            log("PENALTY: step advanced", adv, "for", len(rows), "rows; total ik errors", self.ik_errors)
        return m

    # ---------- geometry ----------
    def pos(self):
        return self.state[:3].copy()

    def rot(self):
        return Rot.from_rotvec(self.state[3:6])

    def grip_open(self):
        return float(self.state[6] + self.state[7])

    def yaw(self):
        """Yaw of the gripper relative to the default top-down orientation Rx(pi)."""
        R = self.rot() * Rot.from_rotvec([np.pi, 0, 0]).inv()
        return float(R.as_euler("xyz")[2])

def target_rot(yaw):
    return Rot.from_euler("z", yaw) * Rot.from_rotvec([np.pi, 0, 0])

def tilt_rot(yaw, plate_angle, theta):
    """Top-down grasp at yaw, then tilt by theta about z x a (a = plate direction): theta>0 = tip down."""
    a = np.array([np.cos(plate_angle), np.sin(plate_angle), 0.0])
    k = np.cross([0, 0, 1.0], a)
    return Rot.from_rotvec(k * theta) * target_rot(yaw)

def move_to(rb, pos, yaw=None, grip=-1.0, tol=0.004, rtol=0.03, max_steps=150,
            vmax=0.02, wmax=0.12, chunk=3, settle=True, rot=None):
    """Closed-loop move of TCP to world pos (and yaw). Returns final error."""
    pos = np.asarray(pos, dtype=float)
    used = 0
    err = 1.0
    rb.penalty_flag = False
    while used < max_steps and not rb.ended:
        if rb.penalty_flag:
            log("move_to abort on penalty; target", np.round(pos, 3))
            break
        p = rb.pos()
        dp = pos - p
        if rot is not None:
            dr = (rot * rb.rot().inv()).as_rotvec()
        elif yaw is not None:
            Re = target_rot(yaw) * rb.rot().inv()
            dr = Re.as_rotvec()
        else:
            dr = np.zeros(3)
        err = float(np.linalg.norm(dp))
        rerr = float(np.linalg.norm(dr))
        if err < tol and rerr < rtol:
            break
        n_lin = int(np.ceil(err / vmax))
        n_rot = int(np.ceil(rerr / wmax))
        n = max(1, n_lin, n_rot)
        k = min(chunk, n)
        rows = []
        for i in range(k):
            # per-step delta: fraction of the remaining error (re-measured each chunk)
            v = dp / n
            w = dr / n
            rows.append([v[0] / 0.05, v[1] / 0.05, v[2] / 0.05,
                         w[0] / 0.5, w[1] / 0.5, w[2] / 0.5, grip])
        rb.step(rows)
        used += k
    return err

def hold(rb, n, grip):
    rows = [[0, 0, 0, 0, 0, 0, grip]] * n
    return rb.step(rows)
