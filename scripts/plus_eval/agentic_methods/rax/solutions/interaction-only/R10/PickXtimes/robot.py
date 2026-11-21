import json, math, time
import numpy as np
from scipy.spatial.transform import Rotation
from perception import parse

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


class Ended(Exception):
    pass


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None
        self.ended = False
        self.task_complete = False
        self.observe()

    def _ingest(self, reply, with_images=True):
        if with_images:
            meta, imgs = parse(reply)
            self.imgs = imgs
        else:
            meta = json.loads(reply["content"][0]["text"])
        self.meta = meta
        self.state = np.array(meta["state"], float)
        self.step_id = meta.get("step_id")
        lim = meta.get("episode_limits", {})
        self.remaining = lim.get("remaining_episode_steps", 10 ** 6)
        if meta.get("episode_ended"):
            self.ended = True
        tr = meta.get("transition") or {}
        if isinstance(tr, dict):
            if tr.get("task_complete"):
                self.task_complete = True
            if tr.get("episode_ended"):
                self.ended = True
        steps = tr.get("steps") if isinstance(tr, dict) else None
        for key in ("steps",):
            v = meta.get(key) or steps
            if isinstance(v, list):
                for s in v:
                    if isinstance(s, dict):
                        if s.get("task_complete"):
                            self.task_complete = True
                        if s.get("episode_ended"):
                            self.ended = True
        if meta.get("task_complete"):
            self.task_complete = True
        return meta

    def observe(self):
        r = self.ctx.call("robomme_observe")
        return self._ingest(r)

    def step(self, rows):
        if self.ended:
            raise Ended()
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        rows = rows[: max(1, min(100, self.remaining))]
        r = self.ctx.call("robomme_step", actions=rows)
        self.last_raw = r
        meta = self._ingest(r)
        if self.ended:
            try:
                txt = r["content"][0]["text"]
                log("EPISODE ENDED meta:", txt[:300])
            except Exception:
                pass
            raise Ended()
        return meta

    @property
    def pos(self):
        return self.state[:3].copy()

    @property
    def grip_width(self):
        return float(self.state[6] + self.state[7])

    def finger_yaw(self):
        R = Rotation.from_rotvec(self.state[3:6]).as_matrix()
        return math.atan2(R[1, 1], R[0, 1])

    def tilt(self):
        R = Rotation.from_rotvec(self.state[3:6]).as_matrix()
        return math.degrees(math.acos(np.clip(-R[2, 2], -1, 1)))

    def tilt_correction(self):
        """World rotvec that brings tool z axis back to pointing straight down."""
        R = Rotation.from_rotvec(self.state[3:6]).as_matrix()
        z = R[:, 2]
        want = np.array([0, 0, -1.0])
        ax = np.cross(z, want)
        s = np.linalg.norm(ax)
        if s < 1e-6:
            return np.zeros(3)
        ang = math.atan2(s, float(np.dot(z, want)))
        return ax / s * ang

    def move_to(self, target, yaw=None, grip=-1.0, tol=0.004, yaw_tol=math.radians(2), speed=0.6,
                max_iter=12, chunk=12):
        target = np.asarray(target, float)
        for it in range(max_iter):
            d = target - self.pos
            dist = np.linalg.norm(d)
            yerr = 0.0
            if yaw is not None:
                yerr = wrap(yaw - self.finger_yaw(), math.pi / 2)
            tc = self.tilt_correction()
            if dist < tol and abs(yerr) < yaw_tol:
                return True
            nmove = int(math.ceil(dist / (speed * 0.05)))
            nrot = int(math.ceil(abs(yerr) / (0.4 * 0.5)))
            n = max(1, min(chunk, max(nmove, nrot)))
            rows = []
            per = d / max(nmove, 1) / 0.05 if nmove > 0 else np.zeros(3)
            prot = yerr / max(nrot, 1) / 0.5 if nrot > 0 else 0.0
            for k in range(n):
                p = per if k < nmove else np.zeros(3)
                rz = prot if k < nrot else 0.0
                rx, ry = (tc[0] / 0.5 * 0.5, tc[1] / 0.5 * 0.5) if k == 0 else (0.0, 0.0)
                rows.append([p[0], p[1], p[2], rx, ry, rz, grip])
            self.step(rows)
        log("move_to: not converged", np.round(target, 3), np.round(self.pos, 3))
        return False

    def hold(self, n, grip):
        return self.step([[0, 0, 0, 0, 0, 0, grip]] * n)


def wrap(a, period):
    """wrap angle a into (-period/2, period/2]."""
    return (a + period / 2) % period - period / 2
