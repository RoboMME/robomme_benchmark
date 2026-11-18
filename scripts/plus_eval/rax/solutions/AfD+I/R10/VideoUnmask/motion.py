import math, json, time
import numpy as np

def aa_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9: return np.eye(3)
    k = v/th; Kx = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + math.sin(th)*Kx + (1-math.cos(th))*Kx@Kx

def R_to_aa(R):
    c = (np.trace(R)-1)/2; c = max(-1.0, min(1.0, c)); th = math.acos(c)
    if th < 1e-7: return np.zeros(3)
    if math.pi - th < 1e-4:
        # near pi
        M = (R + np.eye(3))/2
        k = np.sqrt(np.clip(np.diag(M), 0, None))
        i = int(np.argmax(k)); k = M[:, i]/math.sqrt(max(M[i,i],1e-12))
        return k*th
    w = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/(2*math.sin(th))
    return w*th

def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

DOWN = np.diag([1.0, -1.0, -1.0])  # Rx(pi)

def tool_yaw(state):
    R = aa_to_R(state[3:6]); x = R[:, 0]
    return math.atan2(x[1], x[0])

def wrap_q(a):
    """wrap to [-pi/4, pi/4) (square symmetry)"""
    return (a + math.pi/4) % (math.pi/2) - math.pi/4

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.state = None; self.ended = False; self.complete = False
        self.step_id = 0; self.remaining = None; self.last = None

    def _upd(self, meta):
        self.last = meta
        if "state" in meta: self.state = np.array(meta["state"], float)
        if "step_id" in meta: self.step_id = meta["step_id"]
        lim = meta.get("episode_limits") or {}
        if "remaining_episode_steps" in lim: self.remaining = lim["remaining_episode_steps"]
        if meta.get("episode_ended"): self.ended = True
        for s in meta.get("steps", []) or []:
            if isinstance(s, dict):
                if s.get("task_complete"): self.complete = True
                if s.get("episode_ended"): self.ended = True
        if meta.get("task_complete"): self.complete = True
        tr = meta.get("transition") or {}
        if tr.get("task_complete"): self.complete = True
        if tr.get("episode_ended"): self.ended = True
        for s in tr.get("steps", []) or []:
            if isinstance(s, dict) and s.get("task_complete"): self.complete = True

    def observe(self):
        from percep import parse_reply
        meta, imgs = parse_reply(self.ctx.call("robomme_observe"))
        self._upd(meta); return meta, imgs

    def step(self, rows):
        if self.ended: return None
        rows = [[float(max(-1, min(1, x))) for x in r] for r in rows]
        res = self.ctx.call("robomme_step", actions=rows)
        meta = json.loads(res["content"][0]["text"])
        self._upd(meta)
        return res

    def row_toward(self, target, yaw, grip, gain=1.0, vmax=0.6):
        p = self.state[:3]
        dp = (np.asarray(target) - p)/0.05*gain
        n = np.max(np.abs(dp))
        if n > vmax: dp = dp/n*vmax
        Rt = Rz(yaw) @ DOWN
        R = aa_to_R(self.state[3:6])
        err = R_to_aa(Rt @ R.T)
        dr = err/0.5*gain
        nr = np.max(np.abs(dr))
        if nr > 0.6: dr = dr/nr*0.6
        return list(dp) + list(dr) + [grip]

    def move_to(self, target, yaw, grip, tol=0.004, rtol=0.03, max_steps=80, chunk=2, vmax=0.6):
        target = np.asarray(target, float)
        used = 0
        while used < max_steps and not self.ended:
            p = self.state[:3]
            R = aa_to_R(self.state[3:6]); err = np.linalg.norm(R_to_aa(Rz(yaw) @ DOWN @ R.T))
            dist = np.linalg.norm(target - p)
            if dist < tol and err < rtol: return True
            k = chunk if dist > 0.03 else 1
            rows = []
            # open-loop extrapolation within chunk is avoided: one row recomputed per call when close
            for _ in range(k):
                rows.append(self.row_toward(target, yaw, grip, vmax=vmax))
            self.step(rows); used += k
        return False

    def hold(self, grip, n, yaw=None):
        rows = []
        for _ in range(n):
            if yaw is None: rows.append([0,0,0,0,0,0,grip])
            else: rows.append(self.row_toward(self.state[:3], yaw, grip))
        self.step(rows)
