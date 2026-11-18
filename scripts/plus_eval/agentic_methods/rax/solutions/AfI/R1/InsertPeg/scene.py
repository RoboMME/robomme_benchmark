"""Compare the video's first frame with the execution frame and estimate how the
peg and the box moved (planar SE(2) transforms, identity when unchanged)."""
import numpy as np

from geom import SE2, wrap
from mcpio import log
from vision import table_height, components, comp_stats, half_colors

ABOVE_MIN = 0.006      # m above table counted as object
ABOVE_MAX = 0.30
ROBOT_EXCL = 0.09      # m around the EE treated as robot
SAME_DEPTH = 0.004     # m depth agreement for "unchanged"
SAME_FRAC = 0.80
MOVE_MIN_T = 0.004     # apply a transform (only once depth shows a change) above these
MOVE_MIN_TH = np.deg2rad(2.0)


def _comps(P, tz, ee):
    z = P[..., 2]
    valid = np.all(np.isfinite(P), axis=-1)
    mask = valid & (z > tz + ABOVE_MIN) & (z < tz + ABOVE_MAX)
    if ee is not None:
        near = np.linalg.norm(np.nan_to_num(P, nan=1e3) - np.asarray(ee)[None, None, :], axis=-1) < ROBOT_EXCL
        mask &= ~near
    lab, sizes = components(mask, P, max_jump=0.012, min_size=20)
    return lab, sizes


def _robot_like(P, lab, i, ee, tz):
    """A component reaching high above the table and near the EE column is the arm."""
    pts = P[lab == i]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return True
    if np.percentile(pts[:, 2], 2) > tz + 0.03:
        return True          # not resting on the table: arm link
    if ee is not None:
        if np.min(np.linalg.norm(pts - np.asarray(ee), axis=1)) < ROBOT_EXCL + 0.02 and np.max(pts[:, 2]) > tz + 0.12:
            return True
    return False


def unchanged_fraction(dV, dE, sel):
    a = dV[sel]
    b = dE[sel]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 10:
        return 0.0, 0
    occl = b < a - 0.010          # something (the arm) now in front
    a, b = a[~occl], b[~occl]
    if len(a) < 10:
        return 0.0, 0
    return float(np.mean(np.abs(a - b) < SAME_DEPTH)), len(a)


def analyse(calib, dV, rgbV, dE, rgbE, demo, eeV, eeE):
    """Returns dict with T_peg, T_box (SE2), diagnostics. Never raises on bad data
    in a way the caller cannot catch."""
    out = {"T_peg": SE2(), "T_box": SE2(), "diag": {}}
    PV = calib.backproject(dV)
    PE = calib.backproject(dE)
    tzV = table_height(PV)
    tzE = table_height(PE)
    out["diag"]["tzV"], out["diag"]["tzE"] = tzV, tzE
    if tzV is None or tzE is None:
        out["diag"]["why"] = "no table"
        return out
    out["tz"] = tzV
    # camera static check on table pixels
    tab = np.isfinite(PV[..., 2]) & (np.abs(PV[..., 2] - tzV) < 0.003)
    fr_tab, n_tab = unchanged_fraction(dV, dE, tab)
    out["diag"]["table_same"] = (fr_tab, n_tab)
    if fr_tab < 0.6:
        out["diag"]["why"] = "camera moved? table depth differs"
        log("scene: table depth differs between video and execution (%.2f); assume identity" % fr_tab)
        return out
    hold = demo.grasp_hold() if demo is not None and demo.ok else None
    if hold is None:
        out["diag"]["why"] = "no grasp in demo"
        return out
    g = demo.P[hold["closed"]]
    e_end = demo.P[max(hold["closed"], hold["release"] - 1)]
    out["grasp_pos"] = g
    labV, sizesV = _comps(PV, tzV, eeV)
    nV = len(sizesV)
    # peg: component with most points near grasp xy and below grasp height + 3 cm
    best, best_n = None, 0
    for i in range(nV):
        pts = PV[labV == i]
        if np.percentile(pts[:, 2], 2) > tzV + 0.03:
            continue
        dxy = np.linalg.norm(pts[:, :2] - g[:2], axis=1)
        n = int(np.sum((dxy < 0.03) & (pts[:, 2] < g[2] + 0.04)))
        if n > best_n:
            best, best_n = i, n
    if best is None:
        out["diag"]["why"] = "peg not found in video frame"
        log("scene: peg component not found near grasp", g)
        return out
    peg_sel = labV == best
    stV_peg = comp_stats(PV, rgbV, peg_sel, tzV)
    out["peg_V"] = stV_peg
    # box: nearest other component to the insertion end position, not the robot
    box, box_d = None, 1e9
    for i in range(nV):
        if i == best or _robot_like(PV, labV, i, eeV, tzV):
            continue
        st = comp_stats(PV, rgbV, labV == i, tzV)
        if st is None or st["height"] < 0.015 or st["n"] < 40:
            continue
        pts = PV[labV == i]
        d = float(np.min(np.linalg.norm(pts[:, :2] - e_end[:2], axis=1)))
        if d < box_d:
            box, box_d = i, d
    box_sel = (labV == box) if box is not None and box_d < 0.30 else None
    stV_box = comp_stats(PV, rgbV, box_sel, tzV) if box_sel is not None else None
    out["box_V"] = stV_box
    fr_peg, n_peg = unchanged_fraction(dV, dE, peg_sel)
    out["diag"]["peg_same"] = (fr_peg, n_peg)
    if box_sel is not None:
        fr_box, n_box = unchanged_fraction(dV, dE, box_sel)
    else:
        fr_box, n_box = 1.0, 0
    out["diag"]["box_same"] = (fr_box, n_box)
    log("scene: tz=%.4f peg_same=%.2f (n=%d) box_same=%.2f (n=%d) pegV=%s boxV=%s" % (
        tzV, fr_peg, n_peg, fr_box, n_box, _fmt(stV_peg), _fmt(stV_box)))
    if fr_peg >= SAME_FRAC and fr_box >= SAME_FRAC:
        out["diag"]["why"] = "unchanged"
        return out
    # something moved: match components in the execution frame
    labE, sizesE = _comps(PE, tzE, eeE)
    statsE = []
    for i in range(len(sizesE)):
        if _robot_like(PE, labE, i, eeE, tzE):
            statsE.append(None)
            continue
        statsE.append(comp_stats(PE, rgbE, labE == i, tzE))
    used = set()
    if fr_peg < SAME_FRAC and stV_peg is not None:
        j = _match(stV_peg, statsE, used)
        if j is not None:
            used.add(j)
            T = _se2(stV_peg, statsE[j], PV, rgbV, peg_sel, PE, rgbE, labE == j, elongated=True)
            out["T_peg"] = T
            out["peg_E"] = statsE[j]
            log("scene: peg moved -> match comp %d %s  T=%s" % (j, _fmt(statsE[j]), T))
        else:
            log("scene: peg moved but no match in execution frame")
    if box_sel is not None and fr_box < SAME_FRAC and stV_box is not None:
        j = _match(stV_box, statsE, used)
        if j is not None:
            T = _se2(stV_box, statsE[j], PV, rgbV, box_sel, PE, rgbE, labE == j, elongated=False)
            out["T_box"] = T
            out["box_E"] = statsE[j]
            log("scene: box moved -> match comp %d %s  T=%s" % (j, _fmt(statsE[j]), T))
        else:
            log("scene: box moved but no match")
    out["diag"]["why"] = "moved"
    return out


def _fmt(st):
    if st is None:
        return "None"
    return "{n=%d c=(%.3f,%.3f) h=%.3f L=%.3f W=%.3f yaw=%.1fdeg col=(%d,%d,%d)}" % (
        st["n"], st["c_top"][0], st["c_top"][1], st["height"], st["length"], st["width"],
        np.rad2deg(st["yaw"]), st["color"][0], st["color"][1], st["color"][2])


def _match(sv, statsE, used):
    best, bs = None, 1e9
    for j, se in enumerate(statsE):
        if se is None or j in used:
            continue
        s = (abs(se["height"] - sv["height"]) / 0.01
             + abs(se["length"] - sv["length"]) / max(0.02, 0.25 * sv["length"])
             + abs(se["width"] - sv["width"]) / max(0.02, 0.4 * sv["width"])
             + np.linalg.norm(se["color"] - sv["color"]) / 40.0
             + np.linalg.norm(se["c_top"] - sv["c_top"]) / 0.5)
        if s < bs:
            best, bs = j, s
    if bs > 12:
        return None
    return best


def _se2(sv, se, PV, rgbV, selV, PE, rgbE, selE, elongated):
    th = wrap(se["yaw"] - sv["yaw"])
    elong = sv["length"] > 1.6 * max(sv["width"], 1e-3)
    if elongated or elong:
        # principal axis is defined mod pi: choose with the colour halves, else smallest rotation
        cands = [th, wrap(th + np.pi)]
        hv = half_colors(PV, rgbV, selV, sv) if rgbV is not None else None
        he = half_colors(PE, rgbE, selE, se) if rgbE is not None else None
        chosen = None
        if hv is not None and he is not None:
            asym_v = np.linalg.norm(hv[0] - hv[1])
            if asym_v > 25:
                # th: video +axis maps to exec +axis; th+pi: maps to -axis
                same = np.linalg.norm(hv[0] - he[0]) + np.linalg.norm(hv[1] - he[1])
                flip = np.linalg.norm(hv[0] - he[1]) + np.linalg.norm(hv[1] - he[0])
                chosen = cands[0] if same <= flip else cands[1]
        if chosen is None:
            chosen = min(cands, key=lambda a: abs(a))
        th = chosen
    else:
        # near-square footprint: yaw defined mod pi/2; take smallest
        cands = [wrap(th + k * np.pi / 2) for k in range(4)]
        th = min(cands, key=lambda a: abs(a))
    t = se["c_top"] - sv["c_top"]
    T = SE2(th, t, sv["c_top"])
    if abs(th) < MOVE_MIN_TH and np.linalg.norm(t) < MOVE_MIN_T:
        return SE2()
    return T
