"""Camera geometry: own back-projection validated against robomme_pixel_to_position."""
import itertools
import numpy as np
from mcpio import log, split_result, find_key


def rotvec_to_R(r):
    r = np.asarray(r, float); th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = np.clip(c, -1, 1); th = np.arccos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi: use diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(k)); k = A[:, i] / max(np.sqrt(A[i, i]), 1e-9)
        k = k / np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def parse_p2p(result, n):
    meta, _ = split_result(result)
    lst = None
    for k in ('results', 'points', 'selections', 'positions', 'items'):
        v = find_key(meta, k)
        if isinstance(v, list) and v:
            lst = v; break
    if lst is None:
        for v in meta.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                lst = v; break
    out = []
    for i in range(n):
        p = None
        if lst is not None and i < len(lst) and isinstance(lst[i], dict):
            d = lst[i]
            if d.get('valid') is False:
                out.append(None); continue
            for k in ('position', 'world_position', 'position_m', 'world', 'point', 'xyz', 'world_point', 'position_world'):
                v = d.get(k)
                if isinstance(v, dict):
                    v = [v.get('x'), v.get('y'), v.get('z')]
                try:
                    a = np.asarray(v, float).reshape(-1)
                    if a.size == 3 and np.all(np.isfinite(a)):
                        p = a; break
                except Exception:
                    pass
        out.append(p)
    return out, meta


def tool_points(ctx, step_id, sels, radius=1):
    """sels: list of (cam,u,v). Returns list of xyz or None."""
    res = []
    for i in range(0, len(sels), 32):
        chunk = sels[i:i + 32]
        try:
            r = ctx.call('robomme_pixel_to_position', observation_step_id=int(step_id),
                         selections=[dict(camera=c, u=float(u), v=float(v)) for c, u, v in chunk],
                         depth_window_radius=int(radius))
            pts, meta = parse_p2p(r, len(chunk))
        except Exception as e:
            log('pixel_to_position failed', e)
            pts = [None] * len(chunk)
        res.extend(pts)
    return res


class Camera:
    def __init__(self, K, pose, depth, scale, variant):
        self.K = K; self.pose = pose; self.depth = depth; self.scale = scale; self.v = variant

    def _Rt(self):
        R, t = self.pose['R'], self.pose['t']
        if self.v['xyzw']:
            q = self.pose.get('q')
            if q is not None:
                from mcpio import quat_wxyz_to_R
                R = quat_wxyz_to_R([q[3], q[0], q[1], q[2]])
        if self.v['inv']:
            R2 = R.T; t2 = -R.T @ t
            return R2, t2
        return R, t

    def backproject(self, us, vs, ds):
        us = np.asarray(us, float); vs = np.asarray(vs, float); ds = np.asarray(ds, float)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        off = 0.5 if self.v['half'] else 0.0
        x = (us + off - cx) / fx * ds; y = (vs + off - cy) / fy * ds
        if self.v['gl']:
            pc = np.stack([x, -y, -ds], -1)
        else:
            pc = np.stack([x, y, ds], -1)
        R, t = self._Rt()
        return pc @ R.T + t

    def project(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        R, t = self._Rt()
        pc = (P - t) @ R
        if self.v['gl']:
            d = -pc[:, 2]; x = pc[:, 0]; y = -pc[:, 1]
        else:
            d = pc[:, 2]; x = pc[:, 0]; y = pc[:, 1]
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        off = 0.5 if self.v['half'] else 0.0
        u = x / np.where(np.abs(d) < 1e-9, 1e-9, d) * fx + cx - off
        v = y / np.where(np.abs(d) < 1e-9, 1e-9, d) * fy + cy - off
        return np.stack([u, v, d], -1)

    def depth_at(self, us, vs, r=1):
        H, W = self.depth.shape
        out = []
        for u, v in zip(np.atleast_1d(us), np.atleast_1d(vs)):
            x, y = int(round(u)), int(round(v))
            x0, x1 = max(0, x - r), min(W, x + r + 1); y0, y1 = max(0, y - r), min(H, y + r + 1)
            w = self.depth[y0:y1, x0:x1].reshape(-1)
            w = w[w > 0]
            out.append(np.median(w) * self.scale if len(w) else np.nan)
        return np.array(out)

    def pixels_to_world(self, xs, ys):
        """Exact per-pixel back-projection (no window). Invalid -> nan."""
        xs = np.asarray(xs).astype(int); ys = np.asarray(ys).astype(int)
        d = self.depth[ys, xs] * self.scale
        P = self.backproject(xs, ys, d)
        P[d <= 0] = np.nan
        return P


VARIANTS = [dict(gl=g, half=h, inv=i, xyzw=q) for g, h, i, q in itertools.product((True, False), (False, True), (False, True), (False, True))]


def calibrate(ctx, obs, cam='front', npts=12):
    """Find the projection variant matching the tool. Returns Camera or None (use tool)."""
    if cam not in obs.K or cam not in obs.pose or cam not in obs.depth:
        log('calibrate: missing K/pose/depth for', cam, list(obs.K), list(obs.pose), list(obs.depth))
        return None
    D = obs.depth[cam]
    H, W = D.shape[:2]
    if D.ndim == 3:
        D = D[..., 0]; obs.depth[cam] = D
    rng = np.random.RandomState(0)
    sels = []
    grid = [(u, v) for v in np.linspace(40, H - 30, 5) for u in np.linspace(30, W - 30, 5)]
    rng.shuffle(grid)
    for u, v in grid:
        if D[int(v), int(u)] > 0:
            sels.append((cam, float(int(u)), float(int(v))))
        if len(sels) >= npts:
            break
    if len(sels) < 4:
        log('calibrate: too few valid depth pixels'); return None
    ref = tool_points(ctx, obs.step_id, sels, radius=0)
    pairs = [(s, p) for s, p in zip(sels, ref) if p is not None]
    if len(pairs) < 4:
        log('calibrate: tool returned too few points', ref, '-> using documented OpenGL conversion unvalidated')
        return Camera(obs.K[cam], obs.pose[cam], D, obs.depth_scale.get(cam, 0.001),
                      dict(gl=True, half=False, inv=False, xyzw=False))
    best = None
    for var in VARIANTS:
        c = Camera(obs.K[cam], obs.pose[cam], D, obs.depth_scale.get(cam, 0.001), var)
        errs = []
        for (cm, u, v), p in pairs:
            d = D[int(v), int(u)] * c.scale
            q = c.backproject([u], [v], [d])[0]
            errs.append(np.linalg.norm(q - p))
        e = float(np.median(errs))
        if best is None or e < best[0]:
            best = (e, var, c)
    log('calibrate', cam, 'best err %.4f' % best[0], best[1], 'depth scale', obs.depth_scale.get(cam))
    if best[0] < 0.01:
        return best[2]
    # maybe depth scale wrong: try scale fit
    return None
