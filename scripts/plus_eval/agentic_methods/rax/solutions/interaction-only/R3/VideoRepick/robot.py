"""Robot I/O and motion helpers over the MCP ctx (numpy only)."""
import base64
import io
import json
import math
import numpy as np
from PIL import Image

STEP_POS = 0.05   # m per unit action
STEP_ROT = 0.5    # rad per unit action


def parse(reply):
    meta = {}
    blocks = reply.get("content", [])
    for b in blocks:
        if b.get("type") == "text":
            try:
                meta = json.loads(b["text"])
                break
            except Exception:
                continue
    imgs = {}
    for a in meta.get("attachments", []) or []:
        i = a.get("content_index")
        if i is None or i >= len(blocks) or blocks[i].get("type") != "image":
            continue
        try:
            arr = np.array(Image.open(io.BytesIO(base64.b64decode(blocks[i]["data"]))))
        except Exception:
            continue
        if a["kind"] == "depth":
            arr = arr.astype(np.float64) * 0.001
        imgs[(a["camera"], a["kind"])] = arr
    meta["_imgs"] = imgs
    return meta


def rotvec_to_R(v):
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def R_to_rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(A), 0))
        i = int(np.argmax(k))
        k[i] = math.sqrt(max(A[i, i], 0))
        for j in range(3):
            if j != i:
                k[j] = A[i, j] / k[i]
        k /= np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def down_R(yaw):
    """Gripper pointing down (as reset pose rotvec (pi,0,0)) with extra yaw about world z."""
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return Rz @ np.diag([1.0, -1.0, -1.0])


class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.m = None
        self.ended = False
        self.complete = False
        self.remaining = 10 ** 6
        self.steps = 0
        self.rot_mode = "world"
        self.rot_flipped = False

    def _upd(self, m):
        if m.get("episode_ended") or (m.get("transition") or {}).get("episode_ended"):
            self.ended = True
        tr = m.get("transition") or {}
        if m.get("task_complete") or tr.get("task_complete"):
            self.complete = True
        for s in tr.get("steps", []) or []:
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        lim = m.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if "step_id" in m:
            self.steps = m["step_id"]

    def observe(self):
        m = parse(self.ctx.call("robomme_observe"))
        self._upd(m)
        self.m = m
        return m

    def state(self):
        if self.m is None or "state" not in self.m:
            self.observe()
        return np.array(self.m["state"], float)

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows][:100]
        m = parse(self.ctx.call("robomme_step", actions=rows))
        self._upd(m)
        if "state" in m:
            self.m = m
        else:
            self.observe()
        return m

    def hold(self, n, grip):
        return self.step([[0, 0, 0, 0, 0, 0, grip]] * n)

    def move_to(self, target, grip=-1.0, yaw=0.0, tol=0.004, max_steps=150, gain=1.0,
                vmax=1.0, chunk=4, rot_tol=0.03):
        """Closed-loop move of TCP to world target (xyz) with gripper pointing down + yaw."""
        target = np.asarray(target, float)
        Rt = down_R(yaw)
        used = 0
        er0 = None
        while used < max_steps and not self.ended:
            s = self.state()
            e = target - s[:3]
            Rc = rotvec_to_R(s[3:6])
            if self.rot_mode == "world":
                er = R_to_rotvec(Rt @ Rc.T)
            else:
                er = Rc @ R_to_rotvec(Rc.T @ Rt)
            ner = float(np.linalg.norm(er))
            if er0 is None:
                er0 = ner
            elif ner > er0 + 0.15:
                # rotation diverging: switch convention once, else freeze rotation
                if self.rot_mode == "world" and not self.rot_flipped:
                    self.rot_mode, self.rot_flipped = "body", True
                    self.log("move_to: rotation diverged, switching to body-frame deltas")
                else:
                    self.log("move_to: rotation diverged again, freezing orientation")
                    Rt = Rc
                er0 = None
                continue
            if np.linalg.norm(e) < tol and ner < rot_tol:
                return True
            rows = []
            # plan chunk rows assuming each step achieves its delta
            n = chunk if np.linalg.norm(e) > 0.03 else 2
            for k in range(n):
                d = e * gain / STEP_POS
                nd = np.abs(d).max()
                if nd > vmax:
                    d = d * vmax / nd
                r = er / STEP_ROT
                nr = np.abs(r).max()
                if nr > 0.6:
                    r = r * 0.6 / nr
                rr = r if self.rot_mode == "world" else Rc.T @ r
                rows.append(list(d) + list(rr) + [grip])
                e = e - d * STEP_POS
                er = er - r * STEP_ROT
                if np.linalg.norm(e) < tol * 0.5 and np.linalg.norm(er) < rot_tol * 0.5:
                    break
            self.step(rows)
            used += len(rows)
        s = self.state()
        return bool(np.linalg.norm(target - s[:3]) < 3 * tol)
