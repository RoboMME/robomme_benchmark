"""Container tracking in world XY from the static front camera."""
import numpy as np
from geom import quat_to_R
from percep import white_mask, cluster_xy

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

X_MIN = -0.40       # robot base cluster is at x~-0.57; containers reach x~-0.30 (ep1)
Z_PLANE = 0.06      # container tops are at z~0.07 (ep1)

def white_world_points(rgb, depth, K, pose, zmin=0.012, zmax=0.10):
    wm = white_mask(rgb)
    ys, xs = np.nonzero(wm)
    R = quat_to_R(pose['quaternion_wxyz']); t = np.array(pose['position_m'])
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    rays = np.stack([(xs - cx) / fx, -(ys - cy) / fy, -np.ones_like(xs, float)], 1)
    if depth is not None:
        d = depth[ys, xs]
        ok = d > 0
        P = (rays[ok] * d[ok, None]) @ R.T + t
        P = P[(P[:, 2] > zmin) & (P[:, 2] < zmax)]
    else:
        rw = rays @ R.T
        s = (Z_PLANE - t[2]) / rw[:, 2]
        P = t + rw * s[:, None]
    return P[P[:, 0] > X_MIN]

def detect(rgb, depth, K, pose, exclude=()):
    """exclude: list of (xy, radius) discs to drop (e.g. the gripper while low)."""
    P = white_world_points(rgb, depth, K, pose)
    for xy, rad in exclude:
        P = P[np.linalg.norm(P[:, :2] - np.asarray(xy), axis=1) > rad]
    return cluster_xy(P, cell=0.006, min_pts=15)

class Tracker:
    """Tracks: static buttons and moving containers. Each track: xy, v, label, miss."""
    def __init__(self, gate=0.045):
        self.tracks = []
        self.gate = gate

    def add(self, xy, label, static=False):
        self.tracks.append({'xy': np.array(xy, float), 'v': np.zeros(2), 'label': label,
                            'static': static, 'miss': 0, 'hist': []})

    def update(self, dets, step, dt=1):
        T = self.tracks
        if not T:
            return
        pred = np.array([t['xy'] + (0 if t['static'] else 1) * t['v'] * dt for t in T])
        D = np.array([d['xy'] for d in dets]) if dets else np.zeros((0, 2))
        assigned = {}
        if len(D):
            C = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            Cg = np.where(C < self.gate, C, 1e3)
            if linear_sum_assignment is not None:
                ri, ci = linear_sum_assignment(Cg)
                pairs = [(i, j) for i, j in zip(ri, ci) if Cg[i, j] < 1e3]
            else:
                pairs = []; used_i = set(); used_j = set()
                for k in np.argsort(Cg, axis=None):
                    i, j = divmod(int(k), Cg.shape[1])
                    if Cg[i, j] >= 1e3: break
                    if i in used_i or j in used_j: continue
                    pairs.append((i, j)); used_i.add(i); used_j.add(j)
            assigned = dict(pairs)
        for i, t in enumerate(T):
            if i in assigned:
                new = D[assigned[i]]
                if not t['static']:
                    vel = (new - t['xy']) / dt
                    t['v'] = 0.6 * t['v'] + 0.4 * vel
                    t['xy'] = new
                t['miss'] = 0
            else:
                if not t['static']:
                    t['xy'] = pred[i]
                    t['v'] *= 0.8
                t['miss'] += 1
            t['hist'].append((step, t['xy'].copy(), i in assigned))

    def by_label(self, label):
        for t in self.tracks:
            if t['label'] == label:
                return t
        return None


# ---------------------------------------------------------------- v2: fused multi-camera tracking
WRIST_MIN_CAM_Z = 0.25   # use wrist detections only when the wrist camera is high (whole-table view)
ARM_EXCL_ALONG = 0.13    # hand half-length along finger axis (ep1/reh1: hand spans ~+-0.1 m)
ARM_EXCL_ACROSS = 0.06
ARM_EXCL_EE_Z = 0.25
SPAWN_UNTIL = 20         # containers land by step ~15

def fused_points(m, ims, state, zmax=0.09):
    pts = []
    for cam in ('front', 'wrist'):
        rgb, dep = ims.get((cam, 'rgb')), ims.get((cam, 'depth'))
        if rgb is None or dep is None:
            continue
        pose = m['camera_poses'][cam]
        if cam == 'wrist' and pose['position_m'][2] < WRIST_MIN_CAM_Z:
            continue
        P = white_world_points(rgb, dep, m['camera_intrinsics'][cam]['matrix'], pose, zmax=zmax)
        pts.append(P)
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    ee = np.asarray(state[:3])
    if len(P) and ee[2] < ARM_EXCL_EE_Z:
        from robot import finger_yaw
        yaw = finger_yaw(state)
        u = np.array([np.cos(yaw), np.sin(yaw)]); w = np.array([-u[1], u[0]])
        d = P[:, :2] - ee[:2]
        keep = ~((np.abs(d @ u) < ARM_EXCL_ALONG) & (np.abs(d @ w) < ARM_EXCL_ACROSS))
        P = P[keep]
    return P

class Tracker2(Tracker):
    """Adds growing gates for coasting tracks and re-acquisition by elimination."""
    def __init__(self, gate=0.045, n_containers=None):
        super().__init__(gate)
        self.n_containers = n_containers

    def update(self, dets, step, dt=1):
        T = self.tracks
        if not T:
            return
        pred = np.array([t['xy'] + (0 if t['static'] else 1) * t['v'] * dt for t in T])
        D = np.array([d['xy'] for d in dets]) if dets else np.zeros((0, 2))
        assigned = {}
        if len(D):
            C = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            gates = np.array([self.gate if t['static'] else min(0.10, self.gate + 0.002 * t['miss']) for t in T])
            # coasting tracks pay a small extra cost so fresh tracks win contested detections
            extra = np.array([0.0 if t['static'] else min(0.02, 0.001 * t['miss']) for t in T])
            Cg = np.where(C < gates[:, None], C + extra[:, None], 1e3)
            pairs = _assign(Cg)
            assigned = dict(pairs)
            # elimination: exactly one unmatched moving track and one unmatched detection -> match
            un_t = [i for i, t in enumerate(T) if not t['static'] and i not in assigned and t['miss'] >= 3]
            used = set(assigned.values())
            un_d = [j for j in range(len(D)) if j not in used and dets[j]['n'] >= 40
                    and np.min(np.linalg.norm(pred - D[j], axis=1)) > 0.03]
            if len(un_t) == 1 and len(un_d) == 1:
                assigned[un_t[0]] = un_d[0]
                T[un_t[0]]['v'] = np.zeros(2)
                T[un_t[0]]['reacq'] = T[un_t[0]].get('reacq', 0) + 1
        for i, t in enumerate(T):
            if i in assigned:
                new = D[assigned[i]]
                if not t['static']:
                    vel = (new - t['xy']) / dt
                    if t['miss'] == 0:
                        t['v'] = 0.6 * t['v'] + 0.4 * vel
                    else:
                        t['v'] = np.zeros(2)
                    t['xy'] = new
                t['miss'] = 0
            else:
                if not t['static']:
                    t['xy'] = pred[i]
                    t['v'] *= 0.85
                t['miss'] += 1
            t['hist'].append((step, t['xy'].copy(), i in assigned))

    def final_assign(self, dets):
        """Global assignment of all moving tracks to a complete detection set (large gate)."""
        T = [t for t in self.tracks if not t['static']]
        D = [d for d in dets if d['n'] >= 40]
        S = [t for t in self.tracks if t['static']]
        # drop detections at static (button) positions
        D = [d for d in D if all(np.linalg.norm(d['xy'] - s['xy']) > 0.03 for s in S)]
        if not T or not D:
            return []
        P = np.array([t['xy'] for t in T]); Q = np.array([d['xy'] for d in D])
        C = np.linalg.norm(P[:, None] - Q[None], axis=2)
        w = np.array([1.0 if t['miss'] == 0 else 1.5 for t in T])  # trust fresh tracks more
        pairs = _assign(np.where(C < 0.15, C * w[:, None], 1e3))
        out = []
        for i, j in pairs:
            out.append((T[i]['label'], T[i]['xy'].copy(), Q[j].copy(), float(C[i, j]), T[i]['miss']))
            T[i]['xy'] = Q[j].copy(); T[i]['miss'] = 0
        return out

def _assign(Cg):
    if linear_sum_assignment is not None:
        ri, ci = linear_sum_assignment(Cg)
        return [(int(i), int(j)) for i, j in zip(ri, ci) if Cg[i, j] < 1e3]
    pairs = []; used_i = set(); used_j = set()
    for k in np.argsort(Cg, axis=None):
        i, j = divmod(int(k), Cg.shape[1])
        if Cg[i, j] >= 1e3: break
        if i in used_i or j in used_j: continue
        pairs.append((i, j)); used_i.add(i); used_j.add(j)
    return pairs

def gray_world_points(rgb, depth, K, pose, zmin=0.09, zmax=0.45):
    """Low-saturation (white or gray) points high above the table: pillars, arm links."""
    from percep import hsv
    h, s, v = hsv(rgb)
    m = (s < 0.15) & (v > 0.3) & (depth > 0)
    ys, xs = np.nonzero(m)
    R = quat_to_R(pose['quaternion_wxyz']); t = np.array(pose['position_m'])
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    d = depth[ys, xs]
    P = np.stack([(xs - cx) / fx * d, -(ys - cy) / fy * d, -d], 1) @ R.T + t
    return P[(P[:, 2] > zmin) & (P[:, 2] < zmax)]

def detect_fused(m, ims, state, min_pts=15):
    """Container-like clusters (and buttons) from front + high wrist camera, tall objects rejected."""
    P = fused_points(m, ims, state)
    cl = cluster_xy(P, cell=0.006, min_pts=min_pts)
    tall = []
    for cam in ('front', 'wrist'):
        rgb, dep = ims.get((cam, 'rgb')), ims.get((cam, 'depth'))
        if rgb is None or dep is None:
            continue
        pose = m['camera_poses'][cam]
        if cam == 'wrist' and pose['position_m'][2] < WRIST_MIN_CAM_Z:
            continue
        tall.append(gray_world_points(rgb, dep, m['camera_intrinsics'][cam]['matrix'], pose))
    T = np.concatenate(tall) if tall else np.zeros((0, 3))
    ee = np.asarray(state[:3])
    out = []
    for c in cl:
        if len(T):
            near = np.linalg.norm(T[:, :2] - c['xy'], axis=1) < 0.03
            # ignore tall points that belong to the arm right above its own EE footprint
            if near.sum() > 15:
                c['tall'] = int(near.sum())
                continue
        out.append(c)
    return out
