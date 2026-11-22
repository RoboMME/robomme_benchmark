"""Robot helpers: parse MCP replies, decode images, deproject depth, step."""
import base64, io, json, time
import numpy as np
from PIL import Image

WS = "/workspace"

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def parse(reply):
    meta = json.loads(next(b["text"] for b in reply["content"] if b.get("type") == "text"))
    imgs = {}
    for att in meta.get("attachments", []):
        blk = reply["content"][att["content_index"]]
        if blk.get("type") != "image":
            continue
        im = Image.open(io.BytesIO(base64.b64decode(blk["data"])))
        if att["kind"] == "rgb":
            imgs[(att["camera"], "rgb")] = np.array(im.convert("RGB"))
        else:
            arr = np.array(im).astype(np.float32) * meta["depth"][att["camera"]].get("meters_per_unit", 0.001)
            imgs[(att["camera"], "depth")] = arr
    return meta, imgs

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def deproject(meta, depth, cam, us=None, vs=None):
    """Return world points (H,W,3) or for given pixel arrays."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    if us is None:
        H, W = depth.shape
        vs, us = np.mgrid[0:H, 0:W]
    d = depth[vs, us] if np.ndim(us) else depth[int(vs), int(us)]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(us - cx) * d / fx, -(vs - cy) * d / fy, -d], axis=-1)
    return pc @ R.T + t

def project(meta, cam, pts):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    pc = (np.asarray(pts) - t) @ R  # world->cam
    d = -pc[..., 2]
    u = K[0, 0] * pc[..., 0] / d + K[0, 2]
    v = -K[1, 1] * pc[..., 1] / d + K[1, 2]
    return u, v

class EpisodeEnded(RuntimeError):
    pass


class Robot:
    def __init__(self, ctx, outdir=None):
        self.ctx = ctx
        self.meta = None; self.imgs = None
        self.ended = False; self.complete = False
        self.outdir = outdir
        self.nsaved = 0
        self.on_step = None

    def observe(self):
        self.meta, self.imgs = parse(self.ctx.call("robomme_observe"))
        self._check()
        return self.meta

    def _check(self):
        m = self.meta
        if m.get("episode_ended"):
            self.ended = True
        for s in (m.get("transition") or {}).get("steps", []):
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True

    @property
    def state(self):
        return np.array(self.meta["state"])

    @property
    def step_id(self):
        return self.meta["step_id"]

    def remaining(self):
        return self.meta["episode_limits"]["remaining_episode_steps"]

    def step(self, rows):
        if self.ended:
            raise EpisodeEnded("episode ended")
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        for i in range(0, len(rows), 100):
            self.meta, self.imgs = parse(self.ctx.call("robomme_step", actions=rows[i:i+100]))
            self._check()
            if self.on_step is not None and not self.ended:
                try:
                    self.on_step(self)
                except Exception as e:
                    log("on_step error", repr(e))
            if self.ended:
                break
        return self.meta

    def save(self, tag):
        if not self.outdir:
            return
        try:
            Image.fromarray(np.concatenate([self.imgs[("front", "rgb")], self.imgs[("wrist", "rgb")]], 1)).save(
                f"{self.outdir}/{self.nsaved:03d}_{tag}.png")
            self.nsaved += 1
        except Exception as e:
            log("save fail", e)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def rotvec_to_R(rv):
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx

def R_to_rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-6:
        return np.zeros(3)
    if np.pi - th < 1e-4:
        # near pi
        M = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(M), 0, None))
        i = np.argmax(k); k = M[:, i] / np.sqrt(M[i, i])
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

R_DOWN = rotvec_to_R(np.array([-np.pi, 0, 0]))

def ee_yaw(state):
    """yaw of gripper relative to the home top-down orientation."""
    R = rotvec_to_R(np.array(state[3:6]))
    Rrel = R @ R_DOWN.T
    return float(np.arctan2(Rrel[1, 0], Rrel[0, 0]))

def rot_err(state, yaw):
    """world-frame rotation vector taking current orientation to Rz(yaw)@R_DOWN."""
    R = rotvec_to_R(np.array(state[3:6]))
    Rt = Rz(yaw) @ R_DOWN
    return R_to_rotvec(Rt @ R.T)

def move_to(r, target, yaw=0.0, grip=-1.0, tol=0.004, rtol=0.03, max_steps=120,
            vmax=0.5, gain=1.0, settle=0, kmax=3):
    """Closed-loop: each row commands clipped delta toward target pose."""
    target = np.asarray(target, float)
    n = 0
    while n < max_steps and not r.ended:
        s = r.state
        dp = target - s[:3]
        dr = rot_err(s, yaw)
        if np.linalg.norm(dp) < tol and np.linalg.norm(dr) < rtol:
            break
        a = np.clip(gain * dp / 0.05, -vmax, vmax)
        b = np.clip(dr / 0.5, -0.4, 0.4)
        k = 1 if np.linalg.norm(dp) < 0.03 else kmax
        r.step([list(a) + list(b) + [grip]] * k)
        n += k
    if settle:
        r.step([[0, 0, 0, 0, 0, 0, grip]] * settle)
    return np.linalg.norm(target - r.state[:3])

def hold(r, n, grip):
    r.step([[0, 0, 0, 0, 0, 0, grip]] * n)
