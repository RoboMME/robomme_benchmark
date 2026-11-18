import numpy as np
from geom import deproject_all

COLORS = {
    "stick": (190, 199, 210),
    "pink": (160, 110, 92),
    "teal": (69, 135, 138),
}

def color_mask(rgb, name, tol=18):
    c = np.array(COLORS[name])
    return (np.abs(rgb.astype(int) - c) <= tol).all(-1)

def red_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (r > 120) & (g < 50) & (b < 50)

def purple_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (r > 130) & (b > 170) & (g < 140) & (b - g > 70)

def world_points(meta, imgs, cam, mask, zmax=0.12):
    dep = imgs.get((cam, "depth"))
    if dep is None or mask.sum() == 0:
        return np.zeros((0, 3))
    P = deproject_all(meta, cam, dep)
    pts = P[mask]
    pts = pts[np.isfinite(pts).all(1)]
    return pts[(pts[:, 2] < zmax) & (pts[:, 2] > -0.02)]

def largest_cluster(pts, r=0.04):
    """Keep the largest cluster (greedy single-link in xy)."""
    if len(pts) < 2:
        return pts
    xy = pts[:, :2]
    n = len(xy)
    lab = -np.ones(n, int); k = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack = [i]; lab[i] = k
        while stack:
            j = stack.pop()
            d = np.linalg.norm(xy - xy[j], axis=1)
            nb = np.where((d < r) & (lab < 0))[0]
            lab[nb] = k
            stack.extend(nb.tolist())
        k += 1
    best = np.bincount(lab).argmax()
    return pts[lab == best]

def locate_cube(meta, imgs):
    """Cube centre xy (top-surface based) and top z. Prefer the wrist view."""
    out = {}
    for cam in ("wrist", "front"):
        rgb = imgs.get((cam, "rgb"))
        if rgb is None:
            continue
        pts = world_points(meta, imgs, cam, red_mask(rgb), zmax=0.08)
        pts = largest_cluster(pts, 0.03)
        if len(pts) >= 4:
            ztop = float(np.percentile(pts[:, 2], 90))
            top = pts[pts[:, 2] > ztop - 0.006]
            if len(top) < 3:
                top = pts
            out[cam] = (np.median(top[:, :2], axis=0), ztop, len(pts))
    if not out:
        return None
    if "wrist" in out and out["wrist"][2] >= 10:
        xy, z, n = out["wrist"]
    else:
        xy, z, n = out.get("front", out.get("wrist"))
    return np.array([xy[0], xy[1], z]), out

def locate_target(meta, imgs):
    out = {}
    for cam in ("wrist", "front"):
        rgb = imgs.get((cam, "rgb"))
        if rgb is None:
            continue
        pts = world_points(meta, imgs, cam, purple_mask(rgb), zmax=0.03)
        pts = largest_cluster(pts, 0.04)
        if len(pts) >= 4:
            out[cam] = (np.median(pts[:, :2], axis=0), len(pts))
    if not out:
        return None, out
    if "wrist" in out and out["wrist"][1] >= 10:
        xy = out["wrist"][0]
    else:
        xy = out.get("front", out.get("wrist"))[0]
    return np.array([xy[0], xy[1], 0.0]), out

def locate_hook(meta, imgs, cam="wrist"):
    """Return dict with handle centre H, stick dir u (from handle to end), end E, bar side sign."""
    rgb = imgs.get((cam, "rgb"))
    if rgb is None:
        return None
    teal = world_points(meta, imgs, cam, color_mask(rgb, "teal"), zmax=0.06)
    stick = world_points(meta, imgs, cam, color_mask(rgb, "stick"), zmax=0.04)
    pink = world_points(meta, imgs, cam, color_mask(rgb, "pink", 14), zmax=0.04)
    if len(teal) < 4 or len(stick) < 6:
        return None
    teal = largest_cluster(teal, 0.03)
    H = np.median(teal, axis=0)
    st = stick[np.linalg.norm(stick[:, :2] - H[:2], axis=1) < 0.5]
    if len(st) < 6:
        return None
    rel = st[:, :2] - H[:2]
    # principal direction of stick points relative to handle
    m = rel.mean(0)
    _, _, vt = np.linalg.svd(rel - m)
    u = vt[0]
    if np.dot(u, m) < 0:
        u = -u
    proj = rel @ u
    L_end = float(np.percentile(proj, 98))
    E = H[:2] + u * L_end
    # handle axis from teal points (for grasp yaw); fall back to u
    res = {"H": H, "u": u, "E": E, "L_end": L_end, "n_stick": len(st), "n_teal": len(teal)}
    side = 1.0
    if len(pink) >= 3:
        pk = pink[np.linalg.norm(pink[:, :2] - E, axis=1) < 0.15]
        if len(pk) >= 3:
            w = np.array([-u[1], u[0]])
            lat = (pk[:, :2] - E) @ w
            res["bar_lat"] = float(np.median(lat))
            res["bar_lat_max"] = float(np.percentile(np.abs(lat), 95))
            side = 1.0 if np.median(lat) >= 0 else -1.0
            res["n_pink"] = len(pk)
    res["side"] = side
    return res

# ---------------- generic tool finder (lighting tolerant) ----------------
def teal_mask2(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (g - r > 30) & (b - r > 30) & (np.abs(g - b) < 22) & (r < 130)

def stick_mask2(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (r > 100) & (b - r >= 8) & (b - r <= 32) & (g >= r) & (g <= b + 2)

def pink_mask2(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (r > 140) & (r - g >= 35) & (r - g <= 85) & (g - b >= 6) & (g - b <= 30)

def clusters(pts, r=0.03, min_n=5):
    if len(pts) == 0:
        return []
    xy = pts[:, :2]; n = len(xy)
    lab = -np.ones(n, int); k = 0
    for i in range(n):
        if lab[i] >= 0: continue
        stack = [i]; lab[i] = k
        while stack:
            j = stack.pop()
            d = np.linalg.norm(xy - xy[j], axis=1)
            nb = np.where((d < r) & (lab < 0))[0]
            lab[nb] = k; stack.extend(nb.tolist())
        k += 1
    out = [pts[lab == i] for i in range(k)]
    return sorted([c for c in out if len(c) >= min_n], key=len, reverse=True)

def find_tools(meta, imgs, cams=("wrist", "front"), zmax=0.06):
    """List of tools: dict(H, u, L_end, E, kind ('hook'|'stick'), lat, side, cam)."""
    tools = []
    for cam in cams:
        rgb = imgs.get((cam, "rgb"))
        if rgb is None or imgs.get((cam, "depth")) is None:
            continue
        teal = world_points(meta, imgs, cam, teal_mask2(rgb), zmax=zmax)
        stick = world_points(meta, imgs, cam, stick_mask2(rgb), zmax=0.04)
        stick = stick[stick[:, 2] > 0.002]
        pink = world_points(meta, imgs, cam, pink_mask2(rgb), zmax=0.06)
        pink = pink[pink[:, 2] > 0.003]
        for tc in clusters(teal, 0.025, 6):
            H = np.median(tc, axis=0)
            near = stick[np.linalg.norm(stick[:, :2] - H[:2], axis=1) < 0.12]
            if len(near) < 4:
                continue
            rel = near[:, :2] - H[:2]
            m = rel.mean(0)
            if np.linalg.norm(m) < 1e-3:
                continue
            u = m / np.linalg.norm(m)
            # refine with all stick points near the line
            for _ in range(2):
                relall = stick[:, :2] - H[:2]
                proj = relall @ u
                lat = relall @ np.array([-u[1], u[0]])
                sel = (proj > 0.0) & (proj < 0.6) & (np.abs(lat) < 0.012)
                if sel.sum() < 4:
                    break
                q = relall[sel]
                _, _, vt = np.linalg.svd(q - q.mean(0) * 0, full_matrices=False)
                u2 = vt[0]
                if np.dot(u2, u) < 0: u2 = -u2
                u = u2
            relall = stick[:, :2] - H[:2]
            proj = relall @ u
            lat = relall @ np.array([-u[1], u[0]])
            sel = (proj > 0.0) & (proj < 0.6) & (np.abs(lat) < 0.012)
            if sel.sum() < 4:
                continue
            ps = np.sort(proj[sel])
            # continuity: stop at first gap > 4 cm
            gaps = np.where(np.diff(ps) > 0.04)[0]
            L_stick = float(ps[gaps[0]] if len(gaps) else ps[-1])
            E = H[:2] + u * L_stick
            w = np.array([-u[1], u[0]])
            pk = pink[np.linalg.norm(pink[:, :2] - E, axis=1) < 0.12] if len(pink) else np.zeros((0, 3))
            t = {"cam": cam, "H": H, "u": u, "L_stick": L_stick, "E": E, "n_stick": int(sel.sum()), "n_pink": len(pk)}
            if len(pk) >= 3:
                rel_p = pk[:, :2] - H[:2]
                pl = rel_p @ w; pp = rel_p @ u
                t["pink_lat_med"] = float(np.median(pl))
                t["pink_lat_spread"] = float(np.percentile(pl, 95) - np.percentile(pl, 5))
                t["pink_proj_max"] = float(np.percentile(pp, 97))
                t["pink_z"] = float(np.median(pk[:, 2]))
                t["pink_xy"] = np.median(pk[:, :2], axis=0)
                t["pink_proj_med"] = float(np.median(pp))
                if t["pink_z"] < 0.012 and t["pink_lat_spread"] > 0.045 and abs(t["pink_lat_med"]) < 0.035:
                    t["kind"] = "scoop"
                else:
                    t["kind"] = "hook" if abs(t["pink_lat_med"]) > 0.025 else "stick"
                t["side"] = 1.0 if t["pink_lat_med"] >= 0 else -1.0
                t["L_end"] = max(L_stick, t["pink_proj_max"])
            else:
                t["kind"] = "unknown"; t["side"] = 1.0; t["L_end"] = L_stick
            tools.append(t)
    return tools

def dedup_tools(tools):
    out = []
    for t in tools:
        if any(np.linalg.norm(t["H"][:2] - o["H"][:2]) < 0.04 for o in out):
            continue
        out.append(t)
    return out

# ---------------------------------------------------------------- ball + ramp (board)
def yellow_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (r > 150) & (g > 130) & (b < 100) & (np.abs(r - g) < 50)

def blue_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (b > 150) & (b - r > 50) & (g - r > 25) & (b > g)

def locate_ball(meta, imgs, cams=("wrist", "front")):
    for cam in cams:
        rgb = imgs.get((cam, "rgb"))
        if rgb is None or imgs.get((cam, "depth")) is None:
            continue
        pts = world_points(meta, imgs, cam, yellow_mask(rgb), zmax=0.08)
        pts = largest_cluster(pts, 0.03) if len(pts) else pts
        if len(pts) >= 8:
            return np.median(pts, axis=0), cam
    return None, None

def board_pose(meta, imgs, cams=("front", "wrist")):
    """Inclined board: plane fit; downhill direction g, along coordinate s = (xy-c).g (low edge = s_max)."""
    for cam in cams:
        rgb = imgs.get((cam, "rgb"))
        if rgb is None or imgs.get((cam, "depth")) is None:
            continue
        pts = world_points(meta, imgs, cam, blue_mask(rgb), zmax=0.35)
        pts = pts[pts[:, 2] > 0.004]
        if len(pts) < 150:
            continue
        pts = largest_cluster(pts, 0.03)
        A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
        coef = np.linalg.lstsq(A, pts[:, 2], rcond=None)[0]
        gn = np.linalg.norm(coef[:2])
        if gn < 0.05:
            continue
        g = -coef[:2] / gn
        w = np.array([-g[1], g[0]])
        c = pts[:, :2].mean(0)
        s = (pts[:, :2] - c) @ g
        l = (pts[:, :2] - c) @ w
        return {"cam": cam, "c": c, "coef": coef, "g": g, "w": w, "slope": float(np.arctan(gn)),
                "s_low": float(np.percentile(s, 99)), "s_high": float(np.percentile(s, 1)),
                "l_min": float(np.percentile(l, 2)), "l_max": float(np.percentile(l, 98)), "n": len(pts)}
    return None

def board_z(bd, xy):
    co = bd["coef"]
    return float(co[0] * xy[0] + co[1] * xy[1] + co[2])

def held_tool_angle(meta, imgs, tcp, zmin=0.05, rmax=0.42):
    """Direction (world angle) of a held, lifted tool from the TCP, via stick+pink points above zmin."""
    pts = []
    for cam in ("front", "wrist"):
        rgb = imgs.get((cam, "rgb"))
        if rgb is None or imgs.get((cam, "depth")) is None:
            continue
        for mk in (stick_mask2(rgb), pink_mask2(rgb)):
            q = world_points(meta, imgs, cam, mk, zmax=0.5)
            if len(q):
                d = np.linalg.norm(q[:, :2] - tcp[:2], axis=1)
                q = q[(q[:, 2] > zmin) & (q[:, 2] < tcp[2] + 0.01) & (d > 0.06) & (d < rmax)]
                pts.append(q)
    if not pts:
        return None, 0
    pts = np.concatenate(pts)
    if len(pts) < 30:
        return None, len(pts)
    rel = pts[:, :2] - tcp[:2]
    _, _, vt = np.linalg.svd(rel, full_matrices=False)
    u = vt[0]
    if (rel @ u).mean() < 0:
        u = -u
    return float(np.arctan2(u[1], u[0])), len(pts)

def held_head(meta, imgs, tcp, zmin=0.05, rmax=0.42):
    """Pink head of a held, lifted tool: (head xy world, head z) using the far pink cluster."""
    pts = []
    for cam in ("front", "wrist"):
        rgb = imgs.get((cam, "rgb"))
        if rgb is None or imgs.get((cam, "depth")) is None:
            continue
        q = world_points(meta, imgs, cam, pink_mask2(rgb), zmax=0.5)
        if len(q):
            d = np.linalg.norm(q[:, :2] - tcp[:2], axis=1)
            q = q[(q[:, 2] > zmin) & (q[:, 2] < tcp[2] + 0.01) & (d > 0.08) & (d < rmax)]
            if len(q) >= 15:
                pts.append(q)
                break   # one camera is enough (front preferred)
    if not pts:
        return None
    q = pts[0]
    cl = clusters(q, 0.03, 10)
    if not cl:
        return None
    c = max(cl[:3], key=lambda c: len(c))
    return np.median(c, axis=0)
