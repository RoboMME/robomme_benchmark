"""Helpers: observation decoding, pixel->world, closed-loop moves."""
import base64, io, json, time
import numpy as np
from PIL import Image

LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]])


class Obs:
    def __init__(self, result):
        blocks = result["content"]
        self.meta = json.loads(blocks[0]["text"])
        self.rgb, self.depth = {}, {}
        for att in self.meta.get("attachments", []):
            b = blocks[att["content_index"]]
            img = Image.open(io.BytesIO(base64.b64decode(b["data"])))
            arr = np.array(img)
            if att["kind"] == "rgb":
                self.rgb[att["camera"]] = arr[..., :3]
            else:
                self.depth[att["camera"]] = arr.astype(np.float64) * 0.001
        self.state = np.array(self.meta["state"])
        self.step_id = self.meta["step_id"]
        self.ended = bool(self.meta.get("episode_ended"))
        self.complete = False
        tr = self.meta.get("transition") or {}
        if tr.get("task_complete") or self.meta.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True
        for s in tr.get("steps", []):
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        lim = self.meta.get("episode_limits") or {}
        self.remaining = lim.get("remaining_episode_steps", 10**6)
        self.K = {c: np.array(v["matrix"]) for c, v in self.meta.get("camera_intrinsics", {}).items()}
        self.pose = {c: (np.array(v["position_m"]), quat_to_R(v["quaternion_wxyz"]))
                     for c, v in self.meta.get("camera_poses", {}).items()}

    def pix2world(self, cam, u, v, d=None):
        if d is None:
            d = self.depth[cam][int(round(v)), int(round(u))]
        K = self.K[cam]
        p = np.array([(u - K[0, 2]) * d / K[0, 0], -(v - K[1, 2]) * d / K[1, 1], -d])
        t, R = self.pose[cam]
        return R @ p + t

    def cloud(self, cam):
        """World points for every pixel (H,W,3)."""
        d = self.depth[cam]
        H, W = d.shape
        K = self.K[cam]
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        p = np.stack([(uu - K[0, 2]) * d / K[0, 0], -(vv - K[1, 2]) * d / K[1, 1], -d], -1)
        t, R = self.pose[cam]
        return p @ R.T + t

    def save(self, prefix):
        for c, a in self.rgb.items():
            Image.fromarray(a).save(f"{prefix}_{c}.png")


def observe(ctx):
    return Obs(ctx.call("robomme_observe"))


def step(ctx, rows):
    rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
    return Obs(ctx.call("robomme_step", actions=rows))


def move_to(ctx, target, grip=-1.0, tol=0.004, max_iter=40, chunk=3, gain=1.0, maxv=1.0, obs=None):
    """Closed-loop translation toward target (world m), keep orientation."""
    target = np.asarray(target, float)
    for it in range(max_iter):
        if obs is None:
            obs = observe(ctx)
        if obs.ended:
            return obs, False
        pos = obs.state[:3]
        err = target - pos
        if np.linalg.norm(err) < tol:
            return obs, True
        n = chunk if np.linalg.norm(err) > 0.03 else 1
        rows = []
        for k in range(n):
            dv = np.clip(err * gain / 0.05 / n, -maxv, maxv)
            rows.append([dv[0], dv[1], dv[2], 0, 0, 0, grip])
        obs = step(ctx, rows)
    return obs, False


def hold(ctx, grip, n):
    return step(ctx, [[0, 0, 0, 0, 0, 0, grip]] * n)


from scipy.spatial.transform import Rotation as _Rot


def aa_to_R(v):
    return _Rot.from_rotvec(np.asarray(v, float)).as_matrix()


def R_to_aa(R):
    return _Rot.from_matrix(R).as_rotvec()


def ee_yaw(state):
    R = aa_to_R(state[3:6])
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap90(a):
    """wrap angle to [-pi/4, pi/4)"""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def servo(ctx, target, yaw=None, grip=-1.0, tol=0.003, ytol=0.03, max_iter=40, obs=None):
    """Closed loop on position and (optionally) gripper yaw, keeping the tool pointing down.
    Orientation correction uses the full rotation error to the target R = Rz(yaw)*Rdown."""
    target = np.asarray(target, float)
    for it in range(max_iter):
        if obs is None:
            obs = observe(ctx)
        if obs.ended:
            return obs, False
        pos = obs.state[:3]; Rc = aa_to_R(obs.state[3:6])
        err = target - pos
        rot = np.zeros(3)
        if yaw is not None:
            c, s = np.cos(yaw), np.sin(yaw)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            Rt = Rz @ np.diag([1.0, -1.0, -1.0])
            rot = R_to_aa(Rt @ Rc.T)
        if np.linalg.norm(err) < tol and np.linalg.norm(rot) < ytol:
            return obs, True
        far = np.linalg.norm(err) > 0.03 or np.linalg.norm(rot) > 0.3
        n = 3 if far else 1
        dv = np.clip(err / 0.05 / n, -1, 1)
        dr = np.clip(rot / 0.5 / n * 0.8, -1, 1)
        obs = step(ctx, [[dv[0], dv[1], dv[2], dr[0], dr[1], dr[2], grip]] * n)
    return obs, False
