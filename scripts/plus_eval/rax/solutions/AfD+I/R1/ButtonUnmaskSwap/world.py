"""World model: table, buttons, cubes, bins (tracked)."""
import math
import numpy as np

from obsio import log
from percept import (Grid, Tracker, cam_points, table_height, color_mask, label2d, dilate, hsv)

TOP_BAND = 0.006
COLOR_CLASSES = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'black', 'white']


def classify_colors(h, s, v):
    out = np.full(len(h), '', dtype=object)
    for name in COLOR_CLASSES:
        m = color_mask(h, s, v, name) & (out == '')
        out[m] = name
    return out


def rect_fit(xy):
    """min-area rectangle by angle search. returns (yaw mod 90 deg in rad, w, l, centre)"""
    if len(xy) < 5:
        return None
    c = np.median(xy, 0)
    q = xy - c
    best = None
    for a in np.arange(0, 90, 3.0):
        t = math.radians(a)
        R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
        r = q @ R.T
        lo = np.percentile(r, 3, 0)
        hi = np.percentile(r, 97, 0)
        ar = float(np.prod(hi - lo))
        if best is None or ar < best[0]:
            best = (ar, a)
    a0 = best[1]
    for a in np.arange(a0 - 3, a0 + 3.01, 0.5):
        t = math.radians(a)
        R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
        r = q @ R.T
        lo = np.percentile(r, 3, 0)
        hi = np.percentile(r, 97, 0)
        ar = float(np.prod(hi - lo))
        if ar < best[0]:
            best = (ar, a)
    t = math.radians(best[1])
    R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
    r = q @ R.T
    lo = np.percentile(r, 3, 0)
    hi = np.percentile(r, 97, 0)
    mid = (lo + hi) / 2
    centre = c + mid @ R
    ext = hi - lo
    return t % (math.pi / 2), float(ext[0]), float(ext[1]), centre


class World:
    def __init__(self, calib):
        self.calib = calib
        self.z_table = None
        self.cubes = []      # dict(xy, color, n, ztop)
        self.buttons = []    # dict(xy, cap_top, base_top)
        self.button_mask = None
        self.grid = None
        self.h_bin = None
        self.w_bin = None
        self.tracker = None
        self.track_color = {}  # track index -> colour (targets mostly)
        self.cube_h = 0.02
        self.landed = False
        self.window = None
        self.last_counts = None
        self.wrist_min_dist = 0.12
        self.yaw = {}  # track -> (yaw, step)

    # -------------------------------------------------------------- clouds
    def clouds(self, reply, cams=('front', 'wrist'), step=2):
        out = []
        for cam in cams:
            try:
                cp = cam_points(reply, self.calib, cam, step)
            except Exception as e:
                log('cam_points err', cam, e)
                cp = None
            if cp is not None and cam == 'wrist':
                k = cp.dist > self.wrist_min_dist
                cp = type(cp)(cp.cam, cp.P[k], cp.h[k], cp.s[k], cp.v[k], cp.u[k], cp.vv[k], cp.dist[k])
            if cp is not None:
                out.append(cp)
        return out

    def in_window(self, P):
        if self.window is None:
            return np.ones(len(P), bool)
        x0, x1, y0, y1 = self.window
        return (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)

    def tall_mask(self, cp, h_top):
        """grid of cells containing points well above bin height (robot, pillar)"""
        tall = cp.P[:, 2] > self.z_table + h_top + 0.025
        tall &= self.in_window(cp.P)
        m = self.grid.mask(cp.P[tall, :2])
        return dilate(m, 1)

    # -------------------------------------------------------------- init at step 0
    def init_scene(self, reply):
        cps = self.clouds(reply, step=1)
        if not cps:
            log('init_scene: no clouds')
            return False
        self.z_table = table_height(cps)
        if self.z_table is None:
            allz = np.concatenate([cp.P[:, 2] for cp in cps])
            self.z_table = float(np.percentile(allz, 30))
            log('init_scene: table from percentile', self.z_table)
        zt = self.z_table
        # table-region window from table-coloured points of the front camera
        P = np.concatenate([cp.P for cp in cps])
        tab = np.abs(P[:, 2] - zt) < 0.004
        if tab.sum() > 100:
            xs, ys = P[tab, 0], P[tab, 1]
            self.window = (float(np.percentile(xs, 1)) - 0.05, float(np.percentile(xs, 99)) + 0.05,
                           float(np.percentile(ys, 1)) - 0.05, float(np.percentile(ys, 99)) + 0.05)
        else:
            self.window = (-1.5, 1.5, -1.5, 1.5)
        x0, x1, y0, y1 = self.window
        self.grid = Grid(x0, x1, y0, y1, 0.01)
        fine = Grid(x0, x1, y0, y1, 0.005)
        # low objects: cubes and buttons
        objs = []
        tallm = None
        for cp in cps:
            tm = self.tall_mask(cp, 0.09)
            tallm = tm if tallm is None else (tallm | tm)
        for cp in cps:
            z = cp.P[:, 2] - zt
            low = (z > 0.004) & (z < 0.07) & self.in_window(cp.P)
            low &= ~self.grid.lookup(tallm, cp.P[:, :2])
            objs.append((cp, low))
        allxy = np.concatenate([cp.P[m, :2] for cp, m in objs]) if objs else np.zeros((0, 2))
        if len(allxy) < 10:
            log('init_scene: no low objects')
            return False
        occ = fine.mask(allxy)
        lab, n = label2d(dilate(occ, 1))
        lab = np.where(occ, lab, 0)
        comps = []
        for k in range(1, n + 1):
            ii, jj = np.nonzero(lab == k)
            if len(ii) < 3:
                continue
            comps.append(dict(k=k, area=len(ii) * fine.cell ** 2, xy=fine.centre(ii, jj).mean(0)))
        if not comps:
            return False
        # assign points to components
        pts_by = {c['k']: [] for c in comps}
        for cp, m in objs:
            idx = np.nonzero(m)[0]
            i, j, ok = fine.idx(cp.P[idx, :2])
            labs = np.zeros(len(idx), int)
            labs[ok] = lab[i[ok], j[ok]]
            for c in comps:
                sel = idx[labs == c['k']]
                if len(sel):
                    pts_by[c['k']].append((cp, sel))
        areas = np.array([c['area'] for c in comps])
        med = float(np.median(areas))
        order = np.argsort(-areas)
        # buttons: the two largest components clearly larger than the median cube
        buttons = []
        for o in order[:2]:
            if areas[o] > 1.8 * med and areas[o] > 6e-4:
                buttons.append(comps[o])
        if len(buttons) < 2:
            log('init_scene: buttons ambiguous; areas', np.round(areas * 1e4, 1).tolist())
            buttons = [comps[o] for o in order[:2]]
        bk = set(b['k'] for b in buttons)
        self.buttons = []
        bm = np.zeros_like(occ)
        for b in buttons:
            bm |= (lab == b['k'])
            zs, gray_xy, allp = [], [], []
            for cp, sel in pts_by[b['k']]:
                Pz = cp.P[sel]
                allp.append(Pz)
                g = (cp.v[sel] < 0.78) & (cp.s[sel] < 0.25)
                if g.sum():
                    gray_xy.append(Pz[g])
            allp = np.concatenate(allp)
            cap = np.concatenate(gray_xy) if gray_xy else allp
            cxy = np.median(cap[:, :2], 0) if len(cap) >= 5 else np.median(allp[:, :2], 0)
            near = allp[np.linalg.norm(allp[:, :2] - cxy, axis=1) < 0.012]
            cap_top = float(np.percentile(near[:, 2], 90)) if len(near) else float(np.percentile(allp[:, 2], 95))
            base_top = float(np.percentile(allp[:, 2], 40))
            self.buttons.append(dict(xy=cxy, cap_top=cap_top, base_top=base_top, area=b['area']))
        self.button_mask = dilate(bm, 2)
        self.fine = fine
        # cubes
        self.cubes = []
        for c in comps:
            if c['k'] in bk:
                continue
            cols, zs, xys = [], [], []
            for cp, sel in pts_by[c['k']]:
                cols.append(classify_colors(cp.h[sel], cp.s[sel], cp.v[sel]))
                zs.append(cp.P[sel, 2])
                xys.append(cp.P[sel, :2])
            cols = np.concatenate(cols)
            zs = np.concatenate(zs)
            xys = np.concatenate(xys)
            names, cnt = np.unique(cols[cols != ''], return_counts=True)
            color = names[int(np.argmax(cnt))] if len(names) else 'unknown'
            if c['area'] < 1.5e-4 and len(xys) < 8:
                continue
            self.cubes.append(dict(xy=np.median(xys, 0), color=str(color), n=len(xys),
                                   ztop=float(np.percentile(zs, 95) - zt), area=c['area']))
        if self.cubes:
            self.cube_h = float(np.median([c['ztop'] for c in self.cubes]))
        log('scene: z_table=%.4f window=%s' % (zt, np.round(self.window, 3).tolist()))
        log('scene: buttons', [(np.round(b['xy'], 3).tolist(), round(b['cap_top'] - zt, 4), round(b['base_top'] - zt, 4), round(b['area'] * 1e4, 1)) for b in self.buttons])
        log('scene: cubes', [(c['color'], np.round(c['xy'], 3).tolist(), round(c['ztop'], 3), round(c['area'] * 1e4, 1)) for c in self.cubes])
        return len(self.cubes) >= 3

    # -------------------------------------------------------------- bins
    def visible_tracks(self, reply, cam, preds, margin=3, occ_tol=0.015):
        """bool per predicted bin: is its top face fully inside camera `cam` and unoccluded?"""
        c = self.calib.camera(reply, cam)
        d = self.calib.depth(reply, cam)
        if c is None or d is None or self.h_bin is None:
            return np.ones(len(preds), bool)
        H, W = d.shape
        ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        ring = np.stack([np.cos(ang), np.sin(ang)], 1)
        ring = np.vstack([np.zeros((1, 2)), ring * (0.35 * self.w_bin), ring * (0.62 * self.w_bin)])
        out = np.ones(len(preds), bool)
        for k, q in enumerate(preds):
            X = np.zeros((len(ring), 3))
            X[:, :2] = q + ring
            X[:, 2] = self.z_table + self.h_bin
            u, v, dz = c.project(X)
            if np.any(dz <= 0.02) or np.any(u < margin) or np.any(u > W - 1 - margin) or np.any(v < margin) or np.any(v > H - 1 - margin):
                out[k] = False
                continue
            ui = np.clip(np.round(u).astype(int), 0, W - 1)
            vi = np.clip(np.round(v).astype(int), 0, H - 1)
            obs = d[vi, ui]
            # observed depth along the camera axis vs expected: something in front => occluded
            if np.sum((obs > 0) & (obs < dz - occ_tol)) >= 2:
                out[k] = False
        return out

    def bin_points(self, reply, cams=('front', 'wrist'), h_top=None, extra_excl=None, top_only=True, preds=None):
        """fused XY points belonging to bins (white, in height band, not under tall things)"""
        h = h_top if h_top is not None else (self.h_bin or 0.09)
        zlo = max(0.02, self.cube_h + 0.008)
        if self.h_bin and top_only:
            zlo = max(zlo, self.h_bin - TOP_BAND)  # top face only: unbiased centre from any viewpoint
        out = []
        per = {}
        for cp in self.clouds(reply, cams):
            z = cp.P[:, 2] - self.z_table
            m = (z > zlo) & (z < h + 0.012) & (cp.s < 0.3) & (cp.v > 0.3) & self.in_window(cp.P)
            if m.sum() == 0:
                per[cp.cam] = 0
                continue
            tm = self.tall_mask(cp, h)
            xy = cp.P[m, :2]
            keep = ~self.grid.lookup(tm, xy)
            if self.button_mask is not None and not (self.h_bin and top_only):
                keep &= ~self.fine.lookup(self.button_mask, xy)
            if extra_excl is not None:
                c, r = extra_excl
                keep &= np.linalg.norm(xy - c, axis=1) > r
            if preds is not None and len(preds) and keep.any():
                vis = self.visible_tracks(reply, cp.cam, preds)
                if not vis.all():
                    dd = np.linalg.norm(xy[:, None, :] - preds[None], axis=-1)
                    near = dd.argmin(1)
                    keep &= vis[near] | (dd.min(1) > 0.9 * self.w_bin)
                per[cp.cam + '_hidden'] = int((~vis).sum())
            out.append(xy[keep])
            per[cp.cam] = int(keep.sum())
        self.last_per = per
        if not out:
            return np.zeros((0, 2))
        return np.concatenate(out)

    def landing_check(self, reply):
        xy = self.bin_points(reply, h_top=0.12)
        if not self.cubes or len(xy) == 0:
            return False, 0
        seeds = np.array([c['xy'] for c in self.cubes])
        cnt = 0
        for s in seeds:
            if (np.linalg.norm(xy - s, axis=1) < 0.03).sum() >= 6:
                cnt += 1
        return cnt >= max(2, int(0.6 * len(seeds))), cnt

    def init_tracker(self, reply, step):
        seeds = np.array([c['xy'] for c in self.cubes])
        # measure bin height / width from the landing observation
        cps = self.clouds(reply)
        hs, ws = [], []
        allP = []
        for cp in cps:
            z = cp.P[:, 2] - self.z_table
            m = (z > self.cube_h + 0.008) & (z < 0.15) & (cp.s < 0.3) & (cp.v > 0.3) & self.in_window(cp.P)
            allP.append((cp.cam, cp.P[m]))
        for s in seeds:
            zz = []
            for cam, P in allP:
                d = np.linalg.norm(P[:, :2] - s, axis=1)
                zz.append(P[d < 0.03, 2])
            zz = np.concatenate(zz) if zz else np.zeros(0)
            if len(zz) > 5:
                hs.append(np.percentile(zz, 97) - self.z_table)
        self.h_bin = float(np.median(hs)) if hs else 0.05
        self.h_bin = float(np.clip(self.h_bin, 0.025, 0.12))
        for s in seeds:
            for cam, P in allP:
                if cam != 'wrist':
                    continue
                top = P[(np.linalg.norm(P[:, :2] - s, axis=1) < 0.04) & (P[:, 2] > self.z_table + self.h_bin - 0.006)]
                rf = rect_fit(top[:, :2]) if len(top) > 8 else None
                if rf:
                    ws.append(min(rf[1], rf[2]))
        if not ws:
            for s in seeds:
                for cam, P in allP:
                    top = P[(np.linalg.norm(P[:, :2] - s, axis=1) < 0.04) & (P[:, 2] > self.z_table + self.h_bin - 0.006)]
                    rf = rect_fit(top[:, :2]) if len(top) > 8 else None
                    if rf:
                        ws.append(min(rf[1], rf[2]))
        self.w_bin = float(np.median(ws)) if ws else 0.045
        self.w_bin = float(np.clip(self.w_bin, 0.025, 0.075))
        r = float(np.clip(0.75 * self.w_bin, 0.022, 0.045))
        self.tracker = Tracker(seeds, assign_r=r, min_pts=5)
        self.tracker.merge_d = 0.7 * self.w_bin
        self.track_color = {i: c['color'] for i, c in enumerate(self.cubes)}
        self.landed = True
        xy = self.bin_points(reply)
        self.tracker.update(step, xy, close_d=1.15 * self.w_bin)
        log('tracker init: n=%d h_bin=%.3f w_bin=%.3f assign_r=%.3f' % (len(seeds), self.h_bin, self.w_bin, r))
        return True

    def track(self, reply, step, cams=('front', 'wrist')):
        if not self.landed or self.tracker is None:
            return None
        xy = self.bin_points(reply, cams, preds=self.tracker.predict(step))
        counts = self.tracker.update(step, xy, close_d=1.15 * self.w_bin)
        self.last_counts = counts
        return counts

    # -------------------------------------------------------------- target refinement
    def target_points(self, reply, j, cams=('wrist',), radius=None, centre=None):
        """3-D points of track j's bin from the given cameras (no tall exclusion)."""
        c = self.tracker.p[j] if centre is None else np.asarray(centre, float)
        rr = radius or (0.8 * self.w_bin)
        pts = []
        for cp in self.clouds(reply, cams):
            z = cp.P[:, 2] - self.z_table
            m = (z > max(0.02, 0.45 * self.h_bin)) & (z < self.h_bin + 0.012) & (cp.s < 0.3) & (cp.v > 0.3)
            m &= np.linalg.norm(cp.P[:, :2] - c, axis=1) < rr
            pts.append(cp.P[m])
        return np.concatenate(pts) if pts else np.zeros((0, 3))

    def refine_target(self, reply, j, centre=None):
        """returns (centre_xy, yaw or None, n_top) from wrist top-face points"""
        c0 = self.tracker.p[j] if centre is None else np.asarray(centre, float)
        P = self.target_points(reply, j, centre=c0)
        if len(P) < 10:
            return None
        top = P[P[:, 2] > self.z_table + self.h_bin - 0.007]
        if len(top) < 10:
            return None
        rf = rect_fit(top[:, :2])
        if rf is None:
            return None
        yaw, w, l, centre = rf
        if max(w, l) > 1.3 * self.w_bin + 0.006:
            # probably merged with a neighbour: shrink the window around the current fit and retry once
            c1 = np.median(top[:, :2], 0)
            top2 = top[np.linalg.norm(top[:, :2] - c0, axis=1) < 0.6 * self.w_bin]
            rf2 = rect_fit(top2[:, :2]) if len(top2) >= 10 else None
            if rf2 is None or max(rf2[1], rf2[2]) > 1.3 * self.w_bin + 0.006:
                return (np.median(top2[:, :2], 0) if len(top2) >= 10 else c1), None, len(top)
            yaw, w, l, centre = rf2
        return centre, yaw, len(top)

    def find_color(self, reply, color, near=None, radius=0.06):
        """world XY of visible cube points of colour"""
        pts = []
        for cp in self.clouds(reply):
            z = cp.P[:, 2] - self.z_table
            m = color_mask(cp.h, cp.s, cp.v, color) & (z > 0.003) & (z < 0.06)
            if near is not None:
                m &= np.linalg.norm(cp.P[:, :2] - near, axis=1) < radius
            pts.append(cp.P[m, :2])
        pts = np.concatenate(pts) if pts else np.zeros((0, 2))
        if len(pts) < 4:
            return None, len(pts)
        return np.median(pts, 0), len(pts)

    def visible_colors_near(self, reply, xy, radius=0.05):
        res = {}
        for cp in self.clouds(reply):
            z = cp.P[:, 2] - self.z_table
            m = (z > 0.003) & (z < 0.06) & (np.linalg.norm(cp.P[:, :2] - xy, axis=1) < radius)
            if m.sum() == 0:
                continue
            cols = classify_colors(cp.h[m], cp.s[m], cp.v[m])
            for c in cols:
                if c and c not in ('white',):
                    res[c] = res.get(c, 0) + 1
        return res
