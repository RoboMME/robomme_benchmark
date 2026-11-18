"""Scene parsing from the front RGB-D camera: cube, target, tools, ramp, obstacles."""
import math
from collections import deque
import numpy as np
import perception as P

BASE_XY = np.array([-0.60, 0.0])  # robot base (approx, from front depth: visible base front at x~-0.57)
CUBE_H = 0.021                    # cube half-size


def scene_points(meta, imgs, cam='front'):
    D = imgs[(cam, 'depth')]; c = P.Cam(meta, cam)
    vs, us = np.nonzero(D > 0)
    Pimg = np.zeros(D.shape + (3,)); Pimg[vs, us] = c.backproject(us, vs, D[vs, us])
    return Pimg, D > 0


def plane_points(meta, shape, z0=0.015, cam='front'):
    """Fake geometry for RGB-only demo frames: intersect pixel rays with plane z=z0."""
    c = P.Cam(meta, cam); H, W = shape
    vs, us = np.mgrid[0:H, 0:W]
    p1 = c.backproject(us.ravel(), vs.ravel(), np.ones(H * W))
    o = np.asarray(c.t)
    dirs = p1 - o
    t = (z0 - o[2]) / dirs[:, 2]
    Pp = o + dirs * t[:, None]
    return Pp.reshape(H, W, 3), (t > 0).reshape(H, W)


def color_labels(a):
    return dict(red=P.m_red(a), purple=P.m_purple(a), yellow=P.m_yellow(a), teal=P.m_teal(a),
                blue=P.m_blue(a), salmon=P.m_salmon(a), gray=P.m_gray(a), slate=P.m_slate(a),
                tan=P.m_tan(a))


def components3d(mask, Pimg, max_jump=0.015, min_area=3):
    H, W = mask.shape; seen = np.zeros_like(mask, bool); out = []
    ys0, xs0 = np.nonzero(mask)
    for y0, x0 in zip(ys0, xs0):
        if seen[y0, x0]: continue
        q = deque([(y0, x0)]); seen[y0, x0] = True; ys = []; xs = []
        while q:
            y, x = q.popleft(); ys.append(y); xs.append(x); p = Pimg[y, x]
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    if np.linalg.norm(Pimg[yy, xx] - p) < max_jump:
                        seen[yy, xx] = True; q.append((yy, xx))
        if len(ys) >= min_area: out.append((np.array(ys), np.array(xs)))
    out.sort(key=lambda c: -len(c[0]))
    return out


def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v); return v / n if n > 1e-9 else v * 0


class Scene:
    """Parsed scene. All positions in world metres."""

    def __init__(self, meta, imgs=None, rgb=None, fake=False, log=print):
        self.log = log
        a = rgb if rgb is not None else imgs[('front', 'rgb')]
        self.a = a
        if fake:
            self.Pimg, self.valid = plane_points(meta, a.shape[:2])
        else:
            self.Pimg, self.valid = scene_points(meta, imgs)
        self.fake = fake
        Pm = self.Pimg; x, y, z = Pm[..., 0], Pm[..., 1], Pm[..., 2]
        self.L = color_labels(a)
        near_base = np.hypot(x - BASE_XY[0], y - BASE_XY[1]) < 0.17
        inws = self.valid & (np.abs(x) < 0.9) & (np.abs(y) < 0.9) & ~near_base
        if fake:
            colored = np.zeros(a.shape[:2], bool)
            for k in ('red', 'purple', 'yellow', 'teal', 'blue', 'salmon', 'gray', 'slate', 'tan'):
                colored |= self.L[k]
            self.elev = inws & colored & ~self.L['purple']
        else:
            self.elev = inws & (z > 0.004) & (z < 0.30)
        self.inws = inws
        self.parse()

    # ------------------------------------------------------------------
    def pts(self, mask):
        return self.Pimg[mask & self.valid]

    def parse(self):
        L, Pm = self.L, self.Pimg
        z = Pm[..., 2]
        # cube
        self.cube = None
        red = L['red'] & self.inws
        comps = P.components(red, 4)
        if comps:
            _, ys, xs = comps[0]
            pts = Pm[ys, xs]
            if self.fake:
                c = np.median(pts, 0); c[2] = CUBE_H
            else:
                top = pts[pts[:, 2] > pts[:, 2].max() - 0.006]
                c = np.median(top, 0); c[2] = CUBE_H
                if pts[:, 2].max() < 0.03:  # only side faces visible / low
                    c = np.median(pts, 0); c[2] = CUBE_H
            self.cube = c
        # target: purple rings of the disc; centre = middle of the robust world extent of the cluster
        self.target = None
        pur = L['purple'] & self.valid & (np.abs(Pm[..., 0]) < 0.9) & (Pm[..., 2] < 0.03)
        comps = P.components(pur, 2)
        if comps:
            ys, xs = comps[0][1], comps[0][2]
            c0 = np.median(Pm[ys, xs], 0)
            allp = Pm[pur]
            near = allp[np.hypot(allp[:, 0] - c0[0], allp[:, 1] - c0[1]) < 0.045]
            lo, hi = np.percentile(near[:, :2], 2, 0), np.percentile(near[:, :2], 98, 0)
            t = np.r_[(lo + hi) / 2, 0.0]
            self.target = t
            self.target_extent = hi - lo
        # 3D objects
        self.objs = []
        for ys, xs in components3d(self.elev, Pm, 0.02 if self.fake else 0.015, 4):
            pts = Pm[ys, xs]; cols = {k: float(L[k][ys, xs].mean()) for k in L}
            self.objs.append(dict(ys=ys, xs=xs, pts=pts, cols=cols, n=len(ys),
                                  c=np.median(pts, 0), zmax=float(pts[:, 2].max())))
        self.find_ramp(); self.find_ball(); self.find_teal_tools(); self.find_hammer()

    # ------------------------------------------------------------------
    def find_ramp(self):
        self.ramp = None
        m = self.L['blue'] & self.elev
        if m.sum() < 80: return
        comps = components3d(m, self.Pimg, 0.03, 80)
        if not comps: return
        ys, xs = comps[0]; pts = self.Pimg[ys, xs]
        A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
        coef, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
        for _ in range(2):
            res = np.abs(A @ coef - pts[:, 2]); ok = res < max(0.006, np.percentile(res, 70))
            if ok.sum() < 30: break
            coef, *_ = np.linalg.lstsq(A[ok], pts[ok, 2], rcond=None)
        g = coef[:2]
        self.ramp = dict(pts=pts, coef=coef, grad=g, slope=float(np.linalg.norm(g)), up=unit(g),
                         zmax=float(pts[:, 2].max()), c=np.median(pts, 0))
        # include teal side faces as ramp footprint (for obstacles)
        allm = (self.L['blue'] | self.L['teal']) & self.elev
        foot = []
        for ys2, xs2 in components3d(allm, self.Pimg, 0.03, 30):
            p2 = self.Pimg[ys2, xs2]
            if np.min(np.hypot(*(p2[:, :2] - self.ramp['c'][:2]).T)) < 0.05 and p2[:, 2].max() > 0.05:
                foot.append(p2)
        self.ramp['foot'] = np.concatenate(foot + [pts])

    def find_ball(self):
        self.ball = None
        m = self.L['yellow'] & self.elev
        comps = P.components(m, 6)
        if not comps: return
        _, ys, xs = comps[0]; pts = self.Pimg[ys, xs]
        c = np.median(pts, 0)
        if not self.fake:
            cam = np.array([0.3, 0.0])
            c[:2] = c[:2] + 0.012 * unit(c[:2] - cam)   # visible hemisphere bias
        c[2] = 0.0275 if self.fake else max(0.02, float(pts[:, 2].max()) / 2)
        self.ball = dict(c=c, pts=pts, top=float(pts[:, 2].max()))

    def find_teal_tools(self):
        """Tools with a teal handle: stick, hook, plate. Returns list with axis/tip info."""
        self.tools = []
        L, Pm = self.L, self.Pimg
        tealm = L['teal'] & self.elev
        if self.ramp is not None:
            # drop teal pixels that belong to the ramp's side faces
            rf = self.ramp['foot'][:, :2]
            pass
        comps = components3d(tealm, Pm, 0.02, 5)
        used = np.zeros(L['teal'].shape, bool)
        other = L['red'] | L['yellow'] | L['blue'] | L['purple'] | L['tan']
        tanp = Pm[L['tan'] & self.elev]
        toolmask = self.elev & ~other
        for ys, xs in comps:
            pts = Pm[ys, xs]
            if not self.fake and (pts[:, 2].max() > 0.05 or np.median(pts[:, 2]) > 0.04):
                continue  # ramp side / something tall
            if used[ys[0], xs[0]]: continue
            if len(tanp) >= 5 and np.min(np.hypot(*(tanp[:, :2] - np.median(pts, 0)[:2]).T)) < 0.03:
                continue  # hammer's tip knob, not a teal handle
            # grow the tool body from the handle through teal/gray/salmon pixels
            body = self._grow(toolmask, ys, xs, 0.02 if self.fake else 0.015)
            used |= body
            bp = Pm[body]
            if len(bp) < 10: continue
            h = np.median(pts, 0)
            if not self.fake:
                topz = np.percentile(pts[:, 2], 90)
            else:
                topz = 0.026
            rel = bp[:, :2] - h[:2]
            dist = np.linalg.norm(rel, axis=1)
            far = bp[dist > np.percentile(dist, 90)]
            ax = unit(np.median(far[:, :2], 0) - h[:2])
            s = rel @ ax; l = rel @ np.array([-ax[1], ax[0]])
            Ltot = float(np.percentile(s, 99))
            tip = (s > Ltot - 0.08)
            sal = L['salmon'][body]
            lat = float(np.percentile(l[tip], 95) - np.percentile(l[tip], 5)) if tip.sum() > 3 else 0
            sal_tip = float(sal[tip].mean()) if tip.sum() else 0
            n_sal = int(sal.sum())
            # salmon fill ratio of the tip in a 1 cm grid of tool coordinates
            occ = 0.0
            st = tip & sal
            if st.sum() >= 4:
                gs, gl = np.floor(s[st] / 0.01).astype(int), np.floor(l[st] / 0.01).astype(int)
                cells = set(zip(gs.tolist(), gl.tolist()))
                box = (gs.max() - gs.min() + 1) * (gl.max() - gl.min() + 1)
                occ = len(cells) / box
            # handle axis from teal pixels (more local)
            self.tools.append(dict(handle=h, handle_top=float(topz), axis=ax, length=Ltot, lat=lat,
                                   sal_tip=sal_tip, n_sal=n_sal, occ=occ, n=len(bp), pts=bp,
                                   s=s, l=l, sal=sal, kind=None))
        for t in self.tools:
            t['kind'] = self.tool_kind(t)

    def tool_kind(self, t):
        # plate: big salmon square at the end; hook: wide thin U; stick: small ball tip
        if t['n_sal'] >= 12 and t['occ'] > 0.6 and t['lat'] > 0.04:
            return 'plate'
        if t['lat'] > 0.055:
            return 'hook'
        return 'stick'

    def _grow(self, mask, ys, xs, jump):
        Pm = self.Pimg; H, W = mask.shape
        seen = np.zeros_like(mask, bool); seen[ys, xs] = True
        q = deque(zip(ys.tolist(), xs.tolist()))
        while q:
            y, x = q.popleft(); p = Pm[y, x]
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    if np.linalg.norm(Pm[yy, xx] - p) < jump:
                        seen[yy, xx] = True; q.append((yy, xx))
        return seen

    def find_hammer(self):
        self.hammer = None
        L, Pm = self.L, self.Pimg
        m = L['slate'] & self.elev
        comps = components3d(m, Pm, 0.03, 30)
        if not comps: return
        ys, xs = comps[0]; sp = Pm[ys, xs]
        body = self._grow((L['slate'] | L['tan'] | L['blue'] | L['gray']) & self.elev, ys, xs, 0.02)
        if self.ramp is not None:
            pass
        tanm = L['tan'] & self.elev
        tp = Pm[tanm]
        top = sp[sp[:, 2] > sp[:, 2].max() - 0.015]
        grasp = np.median(top, 0)
        self.hammer = dict(slate=sp, tan=tp, grasp=grasp, ztop=float(sp[:, 2].max()))
        if len(tp) >= 5:
            d = np.linalg.norm(tp[:, :2] - grasp[:2], axis=1)
            tip = tp[d > np.percentile(d, 92)]
            self.hammer['tan_tip'] = np.median(tip, 0)
            self.hammer['tan_z'] = float(np.median(tp[:, 2]))

    # ------------------------------------------------------------------
    def summary(self):
        f = lambda v: None if v is None else np.round(v, 3).tolist()
        s = dict(cube=f(self.cube), target=f(self.target),
                 ball=f(self.ball['c']) if self.ball else None,
                 ramp=dict(c=f(self.ramp['c']), up=f(self.ramp['up']), slope=round(self.ramp['slope'], 3),
                           zmax=round(self.ramp['zmax'], 3)) if self.ramp else None,
                 tools=[dict(kind=t['kind'], handle=f(t['handle']), axis=f(t['axis']), length=round(t['length'], 3),
                             lat=round(t['lat'], 3), sal_tip=round(t['sal_tip'], 2), n_sal=t['n_sal'], occ=round(t['occ'], 2), n=t['n'])
                        for t in self.tools],
                 hammer=dict(grasp=f(self.hammer['grasp']), tip=f(self.hammer.get('tan_tip')))
                 if self.hammer else None)
        return s
