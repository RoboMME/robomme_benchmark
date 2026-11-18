"""v2 tracker: a fixed set of N container tracks (one per step-0 cube), container-top points
fused from the front camera and the wrist camera (when high), no spawning, global
re-association of lost tracks whenever the view is complete."""
import numpy as np
from tracker import white_world_points, _assign
from percep import cluster_xy

TOP_ZMIN = 0.058          # container tops at z~0.073; button caps ~0.034 (excluded)
TOP_ZMAX = 0.095
WRIST_MIN_CAM_Z = 0.25
WRIST_ANY_HEIGHT = True   # low wrist views also used (hand excluded by the rectangle)
ARM_EXCL_EE_Z = 0.30      # hand can show white points in the container band when low
ARM_ALONG, ARM_ACROSS = 0.13, 0.06
MIN_PTS = 15
GATE0, GATE_GROW, GATE_MAX = 0.045, 0.003, 0.10
REACQ_GATE = 0.20


def top_points(m, ims, state, use_wrist=True):
    pts = []
    for cam in ('front', 'wrist'):
        rgb, dep = ims.get((cam, 'rgb')), ims.get((cam, 'depth'))
        if rgb is None or dep is None:
            continue
        pose = m['camera_poses'][cam]
        if cam == 'wrist' and (not use_wrist or (not WRIST_ANY_HEIGHT and pose['position_m'][2] < WRIST_MIN_CAM_Z)):
            continue
        pts.append(white_world_points(rgb, dep, m['camera_intrinsics'][cam]['matrix'], pose,
                                      zmin=TOP_ZMIN, zmax=TOP_ZMAX))
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    ee = np.asarray(state[:3])
    if len(P) and ee[2] < ARM_EXCL_EE_Z:
        from robot import finger_yaw
        yaw = finger_yaw(state)
        u = np.array([np.cos(yaw), np.sin(yaw)]); w = np.array([-u[1], u[0]])
        dd = P[:, :2] - ee[:2]
        P = P[~((np.abs(dd @ u) < ARM_ALONG) & (np.abs(dd @ w) < ARM_ACROSS))]
    return P


def detect_tops(m, ims, state):
    P = top_points(m, ims, state)
    cl = cluster_xy(P, cell=0.005, min_pts=5)
    # merge fragments of one container top (centres closer than 2.2 cm), weighted by size
    cl = sorted(cl, key=lambda c: -c['n'])
    out = []
    for c in cl:
        for o in out:
            if np.linalg.norm(o['xy'] - c['xy']) < 0.022:
                n = o['n'] + c['n']
                o['xy'] = (o['xy'] * o['n'] + c['xy'] * c['n']) / n; o['n'] = n
                break
        else:
            out.append(dict(c))
    out = [c for c in out if c['n'] >= MIN_PTS]
    out = _reject_tall(m, ims, state, out)
    return _split_merged(P, out)


def _reject_tall(m, ims, state, cl):
    """Drop detections under tall low-saturation structure (pillars) away from the arm."""
    from tracker import gray_world_points
    T = []
    for cam in ('front', 'wrist'):
        rgb, dep = ims.get((cam, 'rgb')), ims.get((cam, 'depth'))
        if rgb is None or dep is None:
            continue
        pose = m['camera_poses'][cam]
        if cam == 'wrist' and pose['position_m'][2] < WRIST_MIN_CAM_Z:
            continue
        T.append(gray_world_points(rgb, dep, m['camera_intrinsics'][cam]['matrix'], pose, zmin=0.12, zmax=0.40))
    T = np.concatenate(T) if T else np.zeros((0, 3))
    if not len(T):
        return cl
    ee = np.asarray(state[:3])
    T = T[np.linalg.norm(T[:, :2] - ee[:2], axis=1) > 0.15]   # arm links near the hand
    out = []
    for c in cl:
        k = int((np.linalg.norm(T[:, :2] - c['xy'], axis=1) < 0.03).sum()) if len(T) else 0
        if k > 15:
            continue
        out.append(c)
    return out


def _split_merged(P, cl):
    """Clusters larger than one container top (4 cm) are split with 2-means along the long axis."""
    out = []
    for c in cl:
        if max(c['ext']) < 0.062 or c['n'] < 2 * MIN_PTS:
            out.append(c); continue
        Q = P[np.linalg.norm(P[:, :2] - c['xy'], axis=1) < max(c['ext'])][:, :2]
        if len(Q) < 2 * MIN_PTS:
            out.append(c); continue
        mu = Q.mean(0); X = Q - mu
        ev, evec = np.linalg.eigh(X.T @ X / len(X))
        ax = evec[:, -1]; proj = X @ ax
        a, b = mu + ax * np.percentile(proj, 25), mu + ax * np.percentile(proj, 75)
        for _ in range(8):
            lab = np.linalg.norm(Q - a, axis=1) < np.linalg.norm(Q - b, axis=1)
            if lab.all() or (~lab).all():
                break
            a, b = Q[lab].mean(0), Q[~lab].mean(0)
        if np.linalg.norm(a - b) > 0.03 and lab.sum() >= MIN_PTS and (~lab).sum() >= MIN_PTS:
            out.append({'xy': a, 'n': int(lab.sum()), 'ext': (0.04, 0.04), 'split': True})
            out.append({'xy': b, 'n': int((~lab).sum()), 'ext': (0.04, 0.04), 'split': True})
        else:
            out.append(c)
    return out


class FixedTracker:
    def __init__(self, cubes):
        """cubes: list of (label, xy)."""
        self.tracks = [{'label': l, 'xy': np.array(xy, float), 'v': np.zeros(2), 'miss': 0,
                        'hist': [], 'reacq': 0, 'seen': 0} for l, xy in cubes]
        self.initialised = False

    def init_from(self, dets, step):
        """Containers have just covered the cubes: snap each track to its container."""
        D = np.array([d['xy'] for d in dets]) if dets else np.zeros((0, 2))
        if not len(D):
            return 0
        P = np.array([t['xy'] for t in self.tracks])
        C = np.linalg.norm(P[:, None] - D[None], axis=2)
        pairs = _assign(np.where(C < 0.06, C, 1e3))
        for i, j in pairs:
            self.tracks[i]['xy'] = D[j].copy()
        self.initialised = len(pairs) >= len(self.tracks) - 1
        return len(pairs)

    def update(self, dets, step, dt=1, full_view=False):
        T = self.tracks
        pred = np.array([t['xy'] + t['v'] * dt for t in T])
        D = np.array([d['xy'] for d in dets]) if dets else np.zeros((0, 2))
        assigned = {}
        if len(D):
            C = np.linalg.norm(pred[:, None] - D[None], axis=2)
            gates = np.array([min(GATE_MAX, GATE0 + GATE_GROW * t['miss']) for t in T])
            extra = np.array([min(0.02, 0.001 * t['miss']) for t in T])
            Cg = np.where(C < gates[:, None], C + extra[:, None], 1e3)
            assigned = dict(_assign(Cg))
            # global re-association of lost tracks to orphan detections
            un_t = [i for i in range(len(T)) if i not in assigned and T[i]['miss'] >= 2]
            used = set(assigned.values())
            un_d = [j for j in range(len(D)) if j not in used and dets[j]['n'] >= 2 * MIN_PTS
                    and np.min(np.linalg.norm(pred - D[j], axis=1)) > 0.03]
            if un_t and un_d and (full_view or (len(un_t) == 1 and len(un_d) == 1)):
                C2 = np.linalg.norm(pred[un_t][:, None] - D[un_d][None], axis=2)
                for a, b in _assign(np.where(C2 < REACQ_GATE, C2, 1e3)):
                    i, j = un_t[a], un_d[b]
                    assigned[i] = j
                    T[i]['reacq'] += 1
                    T[i]['v'] = np.zeros(2)
                    T[i]['reacq_log'] = T[i].get('reacq_log', []) + [(step, pred[i].round(3).tolist(), D[j].round(3).tolist())]
        for i, t in enumerate(T):
            if i in assigned:
                new = D[assigned[i]]
                vel = (new - t['xy']) / dt
                t['v'] = 0.6 * t['v'] + 0.4 * vel if t['miss'] == 0 else np.zeros(2)
                sp = np.linalg.norm(t['v'])
                if sp > 0.015:
                    t['v'] *= 0.015 / sp
                t['xy'] = new.copy(); t['miss'] = 0; t['seen'] += 1
            else:
                t['xy'] = pred[i]; t['v'] *= 0.85; t['miss'] += 1
            t['hist'].append((step, t['xy'].copy(), i in assigned))
        return assigned

    def final_assign(self, dets):
        T = self.tracks
        D = [d for d in dets if d['n'] >= 2 * MIN_PTS]
        if not D:
            return []
        P = np.array([t['xy'] for t in T]); Q = np.array([d['xy'] for d in D])
        C = np.linalg.norm(P[:, None] - Q[None], axis=2)
        w = np.array([1.0 if t['miss'] == 0 else 1.5 for t in T])
        out = []
        for i, j in _assign(np.where(C < 0.15, C * w[:, None], 1e3)):
            out.append((T[i]['label'], T[i]['xy'].copy(), Q[j].copy(), float(C[i, j]), T[i]['miss']))
            T[i]['xy'] = Q[j].copy(); T[i]['miss'] = 0
        return out

    def get(self, label, taken=()):
        c = [t for t in self.tracks if t['label'] == label and id(t) not in taken]
        if not c:
            return None
        return min(c, key=lambda t: t['miss'])
