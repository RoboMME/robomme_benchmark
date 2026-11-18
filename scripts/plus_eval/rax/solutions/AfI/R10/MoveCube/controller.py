"""MoveCube controller: classify the video's manner, then reproduce it (hook pull / pick / push)."""
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import motion as M
from motion import move_to, hold, log
import perception as P
from video import analyze, load_frame

OUT = Path("/workspace/output")

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def snapshot(rb):
    meta, imgs = rb.observe()
    return meta, imgs

def scene(rb):
    meta, imgs = snapshot(rb)
    c = P.locate_cube(meta, imgs)
    t, tinfo = P.locate_target(meta, imgs)
    C = None if c is None else c[0]
    log("scene step", rb.step_id, "cube", None if C is None else C.round(4), "target", None if t is None else t.round(4))
    return meta, imgs, C, t

def go_home_view(rb, grip):
    """Move high and away so the front/wrist cameras see the table."""
    p = rb.pos()
    move_to(rb, [p[0], p[1], max(p[2], 0.12)], yaw=None, grip=grip, vmax=0.02, max_steps=40)

BASE = np.array([-0.62, 0.0])

def reach_r(p):
    return float(np.hypot(p[0] - BASE[0], p[1] - BASE[1]))

def reach_ok(p):
    r = reach_r(p)
    return 0.28 < r < 0.78

def safe_z(p, want=0.16):
    """Transit height allowed at xy p (lower when far from the base)."""
    r = reach_r(p)
    if r > 0.72:
        return min(want, 0.08)
    if r > 0.65:
        return min(want, 0.12)
    return want

DEV_EVAL = os.environ.get("MOVECUBE_DEV_EVAL") == "1"

def dev_eval(rb, tag):
    if not DEV_EVAL:
        return
    try:
        r = rb.ctx.call("evaluate")
        m = M._meta(r)
        log("DEV_EVAL", tag, "step", rb.step_id, "task_complete", (m.get("evaluation") or {}).get("task_complete"))
    except Exception as e:
        log("DEV_EVAL failed", repr(e)[:200])

def recover(rb, grip):
    """After an IK penalty: pull back toward the base at the current height."""
    p = rb.pos()
    v = BASE - p[:2]
    q = p[:2] + 0.15 * v
    log("recover to", q.round(3))
    rb.penalty_flag = False
    move_to(rb, [q[0], q[1], max(0.06, p[2])], yaw=None, grip=grip, vmax=0.01, max_steps=40)

def goto(rb, pos, yaw, grip, **kw):
    """move_to with penalty recovery. Returns False if a penalty happened."""
    move_to(rb, pos, yaw=yaw, grip=grip, **kw)
    if rb.penalty_flag:
        recover(rb, grip)
        return False
    return True

# ---------------------------------------------------------------- tools
def tool_summary(t):
    return {k: (np.round(v, 3).tolist() if isinstance(v, np.ndarray) else (round(v, 4) if isinstance(v, float) else v)) for k, v in t.items()}

def choose_tool(meta, imgs, kind):
    tools = P.dedup_tools([t for t in P.find_tools(meta, imgs) if t["kind"] != "unknown"])
    for t in tools:
        log("tool found", tool_summary(t))
    same = [t for t in tools if t["kind"] == kind]
    if same:
        return max(same, key=lambda t: t["n_stick"])
    return None

def grasp_tool(rb, t, tool_dir=None, use_pink=True):
    H, u0 = t["H"], t["u"]
    if use_pink and t.get("kind") == "scoop" and t.get("pink_xy") is not None:
        v = np.asarray(t["pink_xy"]) - H[:2]
        if np.linalg.norm(v) > 0.03:
            u0 = v / np.linalg.norm(v)
    a0 = float(np.arctan2(u0[1], u0[0]))
    cands = [wrap(a0), wrap(a0 - np.pi)]
    if tool_dir is None:
        yg = min(cands, key=lambda y: abs(wrap(y - rb.yaw())))
    else:
        # final yaw when the tool points along tool_dir: tool_dir - (a0 - yg); keep it near neutral
        yg = min(cands, key=lambda y: abs(wrap(tool_dir - (a0 - y))) + 0.5 * abs(y))
    delta = wrap(a0 - yg)
    zg = float(np.clip(H[2] - 0.007, 0.012, 0.03))
    log("grasp yaw", np.degrees(yg), "delta", np.degrees(delta), "zg", zg)
    goto(rb, [H[0], H[1], safe_z(H[:2], 0.14)], yg, -1, vmax=0.02)
    goto(rb, [H[0], H[1], zg], yg, -1, vmax=0.012)
    hold(rb, 12, 1.0)
    width = rb.grip_open()
    log("closed width", width)
    if width < 0.008:
        log("grasp missed")
        hold(rb, 8, -1.0)
        goto(rb, [H[0], H[1], 0.12], yg, -1, vmax=0.02)
        return None
    goto(rb, [H[0], H[1], safe_z(H[:2], 0.16)], yg, 1, vmax=0.012)
    return delta, zg

def best_yaw(alpha, delta, cur):
    return wrap(alpha - delta)

def look_from_above(rb, T, grip=1):
    """Move the wrist camera above the target region and re-measure cube/target."""
    p = rb.pos()
    goto(rb, [p[0], p[1], safe_z(p[:2], 0.20)], None, grip, vmax=0.01, max_steps=60)
    v = T[:2] + 0.25 * (BASE - T[:2]) / max(1e-6, np.linalg.norm(BASE - T[:2])) * 0.0
    q = T[:2] + 0.12 * (BASE - T[:2]) / max(1e-6, np.linalg.norm(BASE - T[:2]))
    goto(rb, [q[0], q[1], safe_z(q, 0.30)], None, grip, vmax=0.015, max_steps=80)
    return scene(rb)

def measure_held(rb):
    """Measured tool geometry in the gripper-yaw frame: axis angle (local), head xy (local), head z below TCP."""
    try:
        meta, imgs = rb.observe()
        p = rb.pos(); y = rb.yaw()
        ang, n = P.held_tool_angle(meta, imgs, p)
        hd = P.held_head(meta, imgs, p)
        if hd is None:
            log("measure_held failed", ang, hd); return None
        if ang is None:
            ang = float(np.arctan2(hd[1] - p[1], hd[0] - p[0]))
        c, s_ = np.cos(-y), np.sin(-y)
        rel = hd[:2] - p[:2]
        hl = np.array([c * rel[0] - s_ * rel[1], s_ * rel[0] + c * rel[1]])
        out = {"axis": wrap(ang - y), "head_ang": float(np.arctan2(hl[1], hl[0])), "head": hl, "head_dz": float(p[2] - hd[2]), "n": n}
        log("held tool: axis local deg", round(np.degrees(out["axis"]), 1), "head local", hl.round(3),
            "|head|", round(float(np.linalg.norm(hl)), 3), "head dz", round(out["head_dz"], 3))
        return out
    except Exception as e:
        log("measure_held err", repr(e)[:150]); return None

def rotz(v, a):
    return np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])

def reorient(rb, ys, grip=1):
    """Lift, then rotate the held tool in place (slowly) before translating: avoids tool slip/collisions."""
    p = rb.pos()
    if abs(wrap(ys - rb.yaw())) < np.radians(10):
        return
    zh = safe_z(p[:2], 0.22)
    goto(rb, [p[0], p[1], max(p[2], zh)], None, grip, vmax=0.012, max_steps=60)
    p = rb.pos()
    goto(rb, [p[0], p[1], p[2]], ys, grip, vmax=0.01, wmax=0.05, max_steps=120)

def correct_delta(rb, delta, tag=""):
    """Measure the held tool's direction (tool lifted) and return the corrected yaw->tool offset."""
    try:
        meta, imgs = rb.observe()
        ang, n = P.held_tool_angle(meta, imgs, rb.pos())
        if ang is None:
            log("tool angle not measured", tag, n); return delta
        d_new = wrap(ang - rb.yaw())
        err = wrap(d_new - delta)
        log("tool angle", tag, "measured", round(np.degrees(ang), 1), "n", n, "delta err deg", round(np.degrees(err), 1))
        if abs(err) < np.radians(35):
            return d_new
    except Exception as e:
        log("correct_delta err", repr(e)[:150])
    return delta

def hook_pull(rb, t, delta, zg, C, T):
    L_end = t["L_end"]
    lat = float(np.clip(abs(t.get("pink_lat_med", 0.04)), 0.025, 0.06))
    side = t["side"]
    mh = measure_held(rb)
    hook_off = float(np.arctan2(side * abs(t.get("pink_lat_med", 0.04)), max(0.1, t.get("pink_proj_med", L_end))))
    if mh is not None:
        mh["axis"] = wrap(mh["head_ang"] - hook_off)
    for attempt in range(3):
        D = T[:2] - C[:2]
        dist = float(np.linalg.norm(D))
        log("pull attempt", attempt, "dist", dist)
        if dist < 0.008:
            break
        us = -D / dist
        ws = side * np.array([-us[1], us[0]])
        s_contact = L_end - 0.026
        gap = 0.03
        s0 = s_contact - gap
        tcp0 = C[:2] - us * s0 - ws * lat
        if mh is not None:
            ys_m = wrap(np.arctan2(us[1], us[0]) - mh["axis"])
            tcp0 = C[:2] + us * (0.024 + gap) - rotz(mh["head"], ys_m)
        tcp1 = tcp0 - us * (gap + dist)
        # keep the pull end within reach (partial pull, re-planned next attempt)
        for _ in range(40):
            if reach_r(tcp1) <= 0.76 or np.linalg.norm(tcp1 - tcp0) < 0.04:
                break
            tcp1 = tcp1 + us * 0.01
        ys = best_yaw(np.arctan2(us[1], us[0]), delta, rb.yaw()) if mh is None else ys_m
        log("pull plan tcp0", tcp0.round(3), "tcp1", tcp1.round(3), "yaw", np.degrees(ys), "r", reach_r(tcp0), reach_r(tcp1))
        zt = min(safe_z(tcp0, 0.16), safe_z(rb.pos()[:2], 0.16))
        reorient(rb, ys)
        goto(rb, [tcp0[0], tcp0[1], zt], ys, 1, vmax=0.015, wmax=0.08, max_steps=200)
        zp = zg + 0.004
        if mh is not None:
            for _ in range(2):
                m2 = measure_held(rb)
                if m2 is None: break
                mh = m2
                mh["axis"] = wrap(mh["head_ang"] - hook_off)
                ys = wrap(np.arctan2(us[1], us[0]) - mh["axis"])
                tcp0 = C[:2] + us * (0.024 + gap) - rotz(mh["head"], ys)
                tcp1 = tcp0 - us * (gap + dist)
                for _k in range(40):
                    if reach_r(tcp1) <= 0.76 or np.linalg.norm(tcp1 - tcp0) < 0.04: break
                    tcp1 = tcp1 + us * 0.01
                goto(rb, [tcp0[0], tcp0[1], zt], ys, 1, vmax=0.01, wmax=0.06, max_steps=80)
            zp = float(np.clip(0.008 + mh["head_dz"], zg - 0.002, zg + 0.02))
            log("pull corrected tcp0", tcp0.round(3), "tcp1", tcp1.round(3), "yaw", round(np.degrees(ys), 1), "zp", round(zp, 3))
        goto(rb, [tcp0[0], tcp0[1], zp], ys, 1, vmax=0.01, max_steps=80)
        dev_eval(rb, "before pull")
        goto(rb, [tcp1[0], tcp1[1], zp], ys, 1, vmax=0.006, max_steps=250, tol=0.003)
        if rb.ended: break
        hold(rb, 6, 1.0)
        dev_eval(rb, "after pull")
        if rb.ended: break
        # release: push the bar away from the cube along +us, then lift slowly
        p = rb.pos()
        goto(rb, [p[0] + us[0] * 0.02, p[1] + us[1] * 0.02, p[2]], ys, 1, vmax=0.004, max_steps=20)
        p = rb.pos()
        goto(rb, [p[0], p[1], zg + 0.05], ys, 1, vmax=0.005, max_steps=40)
        dev_eval(rb, "after release")
        if rb.ended or rb.complete: break
        meta, imgs, C2, T2 = look_from_above(rb, T)
        dev_eval(rb, "after look")
        if rb.ended or C2 is None: break
        if T2 is not None and np.linalg.norm(T2[:2] - T[:2]) < 0.03: T = T2
        C = C2
        log("post-pull err", float(np.linalg.norm(C[:2] - T[:2])))
    return True

def stick_sweep(rb, t, delta, zg, C, T):
    """Push the cube with the ball tip leading (shaft roughly along the push direction)."""
    L = t["L_stick"]
    s_ball = t.get("pink_proj_med", L) if t.get("n_pink", 0) >= 3 else L
    r_ball = float(np.clip(t.get("pink_proj_max", s_ball + 0.014) - s_ball, 0.008, 0.02))
    mh = measure_held(rb)
    for attempt in range(4):
        D = T[:2] - C[:2]
        dist = float(np.linalg.norm(D))
        log("ballpush attempt", attempt, "dist", dist, "s_ball", s_ball, "r_ball", r_ball)
        if dist < 0.006:
            break
        d = D / dist
        gap = 0.03
        back = 0.02 + r_ball + gap          # ball centre distance behind cube centre at start
        seg = dist if attempt > 0 else dist  # full push
        best = None
        for phi in np.radians([0, 25, -25, 45, -45, 60, -60]):
            u = np.array([np.cos(phi) * d[0] - np.sin(phi) * d[1], np.sin(phi) * d[0] + np.cos(phi) * d[1]])
            B0 = C[:2] - d * back
            if mh is not None:
                ys_m = wrap(np.arctan2(u[1], u[0]) - mh["head_ang"])
                G0 = B0 - rotz(mh["head"], ys_m)
            else:
                G0 = B0 - s_ball * u
            G1 = G0 + d * (gap + seg - 0.002)
            ys = best_yaw(np.arctan2(u[1], u[0]), delta, rb.yaw()) if mh is None else ys_m
            feas = reach_ok(G0) and reach_ok(G1) and min(reach_r(G0), reach_r(G1)) > 0.30
            cost = 0.6 * abs(wrap(ys - rb.yaw())) + abs(phi) * 0.8 + (0 if feas else 10) \
                + 3.0 * max(0.0, max(reach_r(G0), reach_r(G1)) - 0.70) + 2.0 * max(0.0, abs(ys) - 2.3)
            if best is None or cost < best[0]:
                best = (cost, G0, G1, ys, phi)
        _, G0, G1, ys, phi = best
        log("ballpush plan G0", G0.round(3), "G1", G1.round(3), "yaw", np.degrees(ys), "phi", np.degrees(phi),
            "r", reach_r(G0), reach_r(G1))
        zt = min(safe_z(G0, 0.16), safe_z(rb.pos()[:2], 0.16))
        reorient(rb, ys)
        goto(rb, [G0[0], G0[1], zt], ys, 1, vmax=0.015, wmax=0.08, max_steps=200)
        zp = zg + 0.005
        if mh is not None:
            u = rotz(np.array([np.cos(mh["head_ang"]), np.sin(mh["head_ang"])]), ys)
            for _ in range(2):
                m2 = measure_held(rb)
                if m2 is None: break
                mh = m2
                ys = wrap(np.arctan2(u[1], u[0]) - mh["head_ang"])
                B0 = C[:2] - d * back
                G0 = B0 - rotz(mh["head"], ys)
                G1 = G0 + d * (gap + seg - 0.002)
                goto(rb, [G0[0], G0[1], zt], ys, 1, vmax=0.01, wmax=0.06, max_steps=80)
            zp = float(np.clip(0.03 + mh["head_dz"], zg - 0.002, zg + 0.02))
            log("ball push corrected G0", G0.round(3), "G1", G1.round(3), "yaw", round(np.degrees(ys), 1), "zp", round(zp, 3))
        goto(rb, [G0[0], G0[1], zp], ys, 1, vmax=0.01, max_steps=80)
        dev_eval(rb, "before push")
        goto(rb, [G1[0], G1[1], zp], ys, 1, vmax=0.005, max_steps=300, tol=0.003)
        if rb.ended: break
        hold(rb, 6, 1.0)
        dev_eval(rb, "after push")
        if rb.ended: break
        p = rb.pos()
        goto(rb, [p[0] - d[0] * 0.02, p[1] - d[1] * 0.02, p[2]], ys, 1, vmax=0.004, max_steps=15)
        p = rb.pos()
        goto(rb, [p[0], p[1], zg + 0.05], ys, 1, vmax=0.005, max_steps=40)
        dev_eval(rb, "after lift")
        if rb.ended or rb.complete: break
        meta, imgs, C2, T2 = look_from_above(rb, T)
        dev_eval(rb, "after look")
        if rb.ended or rb.complete or C2 is None: break
        if T2 is not None and np.linalg.norm(T2[:2] - T[:2]) < 0.03: T = T2
        C = C2
        log("post-push err", float(np.linalg.norm(C[:2] - T[:2])))
    return True

def plate_offset(rb, fallback):
    """Measure the held scoop plate relative to the TCP from the wrist camera."""
    meta, imgs = rb.observe()
    p = rb.pos()
    try:
        pk = P.world_points(meta, imgs, "wrist", P.pink_mask2(imgs[("wrist", "rgb")]), zmax=0.5)
        pk = pk[np.linalg.norm(pk[:, :2] - p[:2], axis=1) < 0.2]
        if len(pk) > 200:
            pc = np.median(pk, 0)
            off = pc[:2] - p[:2]
            dp = float(np.linalg.norm(off)); dz = float(p[2] - pc[2])
            if 0.05 < dp < 0.16 and 0.0 < dz < 0.04:
                return dp, dz, wrap(np.arctan2(off[1], off[0]) - rb.yaw())
    except Exception as e:
        log("plate_offset err", repr(e)[:120])
    return fallback

def cube_top(rb):
    meta, imgs = rb.observe()
    out = None
    for cam in ("wrist", "front"):
        try:
            pts = P.world_points(meta, imgs, cam, P.red_mask(imgs[(cam, "rgb")]), zmax=0.3)
            pts = P.largest_cluster(pts, 0.03)
            if len(pts) >= 10:
                xy = np.median(pts[:, :2], 0); zt = float(np.percentile(pts[:, 2], 97))
                if out is None or cam == "wrist":
                    out = (xy, zt, cam)
                    if cam == "wrist": break
        except Exception:
            pass
    return out

def scoop_carry(rb, t, delta, zg, C, T):
    fb = (float(t.get("pink_proj_med", 0.1)), 0.016, delta)
    dp, dz, dpl = plate_offset(rb, fb)
    log("plate offset dp", dp, "dz", dz, "delta_plate", np.degrees(dpl))
    half = 0.045
    th = np.radians(6.5)
    radial = np.arctan2(C[1] - BASE[1], C[0] - BASE[0])
    for attempt in range(2):
        best = None
        for a in radial + np.radians([0, -30, 30, -55, 55, -80, 80]):
            ua = np.array([np.cos(a), np.sin(a)])
            S = C[:2] - ua * (dp + half + 0.02 + 0.03)
            E = C[:2] - ua * (dp - 0.01) + ua * 0.06
            y = wrap(a - dpl)
            feas = reach_ok(S) and reach_ok(E) and reach_r(S) > 0.32
            cost = 0.5 * abs(y) + 0.4 * abs(wrap(a - radial)) + (0 if feas else 10) + 3 * max(0, reach_r(E) - 0.68)
            if best is None or cost < best[0]:
                best = (cost, a, ua, S, y)
        _, a, ua, S, y = best
        zs = 0.006 + dz + (dp + half) * np.sin(th)
        R = M.tilt_rot(y, a, th)
        log("scoop plan a", np.degrees(a), "yaw", np.degrees(y), "S", S.round(3), "zs", zs)
        p = rb.pos()
        reorient(rb, y)
        goto(rb, [p[0], p[1], max(p[2], 0.12)], y, 1, vmax=0.01, wmax=0.08, max_steps=120)
        goto(rb, [S[0], S[1], 0.12], y, 1, vmax=0.015, wmax=0.08, max_steps=200)
        move_to(rb, [S[0], S[1], zs + 0.03], rot=R, grip=1, vmax=0.01, max_steps=80)
        move_to(rb, [S[0], S[1], zs], rot=R, grip=1, vmax=0.004, max_steps=40)
        onboard = False
        travelled = 0.0
        while travelled < 0.26 and not rb.ended:
            g = rb.pos()[:2] + ua * 0.025
            if not reach_ok(g) or reach_r(g) > 0.74:
                log("scoop slide reach limit"); break
            move_to(rb, [g[0], g[1], zs], rot=R, grip=1, vmax=0.005, max_steps=30, tol=0.003)
            travelled += 0.025
            ct = cube_top(rb)
            if ct is None:
                continue
            rel = ct[0] - rb.pos()[:2]
            along = float(rel @ ua)
            log("slide", round(travelled, 3), "cube top", round(ct[1], 4), ct[2], "along", round(along, 3))
            if ct[1] > 0.0465 and along < dp + half - 0.015:
                onboard = True
                if along < dp + 0.01:
                    break
            elif along < dp - 0.03:
                break
        dev_eval(rb, "after slide")
        if rb.ended: return True
        if not onboard:
            log("scoop: cube not on plate")
            p = rb.pos()
            goto(rb, [p[0] - ua[0] * 0.04, p[1] - ua[1] * 0.04, zs + 0.04], y, 1, vmax=0.006, max_steps=40)
            meta, imgs, C2, T2 = look_from_above(rb, T)
            if C2 is None: return False
            C = C2
            if T2 is not None and np.linalg.norm(T2[:2] - T[:2]) < 0.03: T = T2
            continue
        # tilt tip up a little and lift
        p = rb.pos()
        Rup = M.tilt_rot(y, a, np.radians(-4))
        move_to(rb, [p[0], p[1], zs + 0.015], rot=Rup, grip=1, vmax=0.004, wmax=0.03, max_steps=40)
        p = rb.pos()
        move_to(rb, [p[0], p[1], 0.12], rot=Rup, grip=1, vmax=0.006, max_steps=60)
        ct = cube_top(rb)
        rel_along = dp
        if ct is not None:
            rel_along = float((ct[0] - rb.pos()[:2]) @ ua)
            log("lifted cube top", round(ct[1], 4), "along", round(rel_along, 3))
        tip_to_cube = max(0.0, dp + half - rel_along)
        # TCP so that the cube slides off the tip and lands on the target
        Q = T[:2] - ua * (rel_along + tip_to_cube + 0.015)
        log("carry to", Q.round(3), "tip_to_cube", tip_to_cube)
        move_to(rb, [Q[0], Q[1], 0.12], rot=Rup, grip=1, vmax=0.008, max_steps=120)
        move_to(rb, [Q[0], Q[1], 0.07], rot=Rup, grip=1, vmax=0.005, max_steps=40)
        dev_eval(rb, "before drop")
        for k, deg in enumerate([0, 6, 12, 18, 24, 30]):
            move_to(rb, [Q[0], Q[1], 0.07 + 0.002 * k], rot=M.tilt_rot(y, a, np.radians(deg)), grip=1,
                    vmax=0.004, wmax=0.03, max_steps=20)
        hold(rb, 15, 1.0)
        dev_eval(rb, "after drop")
        if rb.ended or rb.complete: return True
        p = rb.pos()
        move_to(rb, [p[0] - ua[0] * 0.05, p[1] - ua[1] * 0.05, 0.14], yaw=y, grip=1, vmax=0.006, max_steps=60)
        meta, imgs, C2, T2 = look_from_above(rb, T)
        dev_eval(rb, "after look")
        if C2 is not None:
            log("scoop final err", float(np.linalg.norm(C2[:2] - T[:2])))
        return True
    return True
def tool_strategy(rb, meta, imgs, C, T, kind):
    t = choose_tool(meta, imgs, kind)
    if t is None:
        # look from a higher, more central pose
        move_to(rb, [-0.15, 0.0, 0.45], yaw=0.0, grip=-1, vmax=0.03)
        meta, imgs, C2, T2 = scene(rb)
        t = choose_tool(meta, imgs, kind)
        if C2 is not None: C = C2
        if T2 is not None: T = T2
    if t is None:
        log("no tool of kind", kind); return False
    log("chosen tool", tool_summary(t))
    if kind == "hook":
        hint = float(np.arctan2(C[1] - T[1], C[0] - T[0]))
    elif kind == "scoop":
        hint = float(np.arctan2(C[1] - BASE[1], C[0] - BASE[0]))
    else:
        hint = float(np.arctan2(T[1] - C[1], T[0] - C[0]))
    g = grasp_tool(rb, t, hint)
    if g is None and not rb.ended:
        log("grasp retry from a high view")
        goto(rb, [-0.06, 0.0, 0.45], 0.0, -1, vmax=0.03)
        meta, imgs, C2, T2 = scene(rb)
        t2 = choose_tool(meta, imgs, kind)
        if C2 is not None: C = C2
        if T2 is not None: T = T2
        if t2 is not None:
            t = t2
            g = grasp_tool(rb, t, hint, use_pink=False)
    if g is None:
        return False
    delta, zg = g
    if kind == "hook":
        return hook_pull(rb, t, delta, zg, C, T)
    if kind == "scoop":
        return scoop_carry(rb, t, delta, zg, C, T)
    return stick_sweep(rb, t, delta, zg, C, T)

# ---------------------------------------------------------------- ramp (roll the ball down the board)
RAMP_K = 20.1      # d = K * h * cos(slope)^2 - D0  (fit: video 37deg h.0275->.179, reh3 18.9deg h.0324->.408)
RAMP_D0 = 0.175

def ramp_strategy(rb, meta, imgs, C, T, vinfo):
    vr = vinfo.get("ramp") or {}
    d0 = RAMP_D0
    h_last = None
    shots = 0
    for attempt in range(4):
        if shots >= 3:
            break
        bxyz, bcam = P.locate_ball(meta, imgs)
        bd = P.board_pose(meta, imgs)
        if bxyz is None or bd is None:
            move_to(rb, [-0.15, 0.0, 0.45], yaw=0.0, grip=-1, vmax=0.03)
            meta, imgs, C2, T2 = scene(rb)
            if C2 is not None: C = C2
            if T2 is not None: T = T2
            bxyz, bcam = P.locate_ball(meta, imgs)
            bd = P.board_pose(meta, imgs)
        if bxyz is None or bd is None:
            log("ramp: ball or board missing", bxyz, bd is not None); return False
        log("ball", bxyz.round(3), bcam, "board c", bd["c"].round(3), "g", bd["g"].round(3), "slope", np.degrees(bd["slope"]),
            "s_low", round(bd["s_low"], 3), "s_high", round(bd["s_high"], 3), "lat", round(bd["l_min"], 3), round(bd["l_max"], 3))
        if shots > 0:
            if np.linalg.norm(bxyz[:2] - C[:2]) < 0.065 or not reach_ok(bxyz[:2]):
                log("ramp: ball not retrievable", bxyz.round(3)); return True
        # grasp the ball from above
        y = 0.0
        goto(rb, [bxyz[0], bxyz[1], safe_z(bxyz[:2], 0.12)], y, -1, vmax=0.02)
        meta2, imgs2 = rb.observe()
        b2, _ = P.locate_ball(meta2, imgs2, cams=("wrist",))
        if b2 is not None and np.linalg.norm(b2[:2] - bxyz[:2]) < 0.04:
            bxyz = b2
            goto(rb, [bxyz[0], bxyz[1], safe_z(bxyz[:2], 0.10)], y, -1, vmax=0.01)
        goto(rb, [bxyz[0], bxyz[1], 0.021], y, -1, vmax=0.012)
        hold(rb, 14, 1.0)
        w = rb.grip_open()
        log("ball grasp width", w)
        if w < 0.02:
            hold(rb, 8, -1.0)
            goto(rb, [bxyz[0], bxyz[1], 0.12], y, -1, vmax=0.02)
            meta, imgs, C2, T2 = scene(rb)
            if C2 is not None: C = C2
            continue
        # release point on the board
        g, wv = bd["g"], bd["w"]
        d_now = float(np.linalg.norm(T[:2] - C[:2]))
        c2 = float(np.cos(bd["slope"]) ** 2)
        h_t = (d_now + d0) / (RAMP_K * max(c2, 0.3))
        if shots == 0:
            h_t *= 0.92          # first shot slightly conservative; undershoot is recoverable
        h_t = float(np.clip(h_t, 0.010, 0.10))
        s = bd["s_low"] - h_t / max(np.tan(bd["slope"]), 0.1)
        log("ramp height target", h_t, "d_now", d_now, "d0", d0, "cos2", c2)
        s = float(np.clip(s, bd["s_high"] + 0.03, bd["s_low"] - 0.02))
        h_used = float(np.tan(bd["slope"]) * (bd["s_low"] - s))
        l = float((C[:2] - bd["c"]) @ wv)
        l = float(np.clip(l, bd["l_min"] + 0.03, bd["l_max"] - 0.03))
        R = bd["c"] + g * s + wv * l
        zr = P.board_z(bd, R) + float(np.clip(vr.get("dz_surface", 0.023), 0.021, 0.03))
        ztop = max(0.16, P.board_z(bd, bd["c"] + g * bd["s_high"]) + 0.08)
        log("ramp release", R.round(3), "z", round(zr, 3), "h_used", round(h_used, 4), "transit", round(ztop, 3),
            "cube", C.round(3), "target", T.round(3))
        p = rb.pos()
        goto(rb, [p[0], p[1], ztop], y, 1, vmax=0.012)
        goto(rb, [R[0], R[1], ztop], y, 1, vmax=0.015)
        goto(rb, [R[0], R[1], zr + 0.02], y, 1, vmax=0.01)
        goto(rb, [R[0], R[1], zr], y, 1, vmax=0.004, max_steps=40)
        hold(rb, 4, 1.0)
        dev_eval(rb, "before release")
        hold(rb, 10, -1.0)
        shots += 1
        p = rb.pos()
        goto(rb, [p[0], p[1], ztop], y, -1, vmax=0.01, max_steps=60)
        hold(rb, 40, -1.0)
        dev_eval(rb, "after roll")
        if rb.ended or rb.complete: return True
        goto(rb, [-0.06, 0.0, 0.45], 0.0, -1, vmax=0.03)
        meta, imgs, C2, T2 = scene(rb)
        if C2 is None: return True
        if T2 is not None and np.linalg.norm(T2[:2] - T[:2]) < 0.03: T = T2
        moved = float(np.linalg.norm(C2[:2] - C[:2]))
        err = float(np.linalg.norm(C2[:2] - T[:2]))
        log("ramp result cube", C2.round(3), "moved", moved, "err", err, "shot", shots)
        if moved > 0.03:
            # recalibrate the offset from the observed shot
            d0 = float(np.clip(RAMP_K * h_used * c2 - moved, 0.0, 0.5))
            dirm = (C2[:2] - C[:2]) / moved
            remaining = float((T[:2] - C2[:2]) @ dirm)
            log("ramp recal d0", d0, "remaining", remaining)
            C = C2
            if err < 0.03 or remaining < 0.03:
                return True
        else:
            C = C2
            d0 = d0 + 0.08    # too weak or missed: release higher next time
        if rb.step_id > 1000:
            return True
    return True

# ---------------------------------------------------------------- pick
def cube_yaw(meta, imgs):
    rgb = imgs.get(("wrist", "rgb"))
    pts = P.world_points(meta, imgs, "wrist", P.red_mask(rgb), zmax=0.08)
    pts = P.largest_cluster(pts, 0.03)
    if len(pts) < 8:
        return 0.0
    zt = np.percentile(pts[:, 2], 90)
    xy = pts[pts[:, 2] > zt - 0.006][:, :2]
    xy = xy - xy.mean(0)
    best, ba = 1e9, 0.0
    for a in np.radians(np.arange(0, 90, 3)):
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        q = xy @ R.T
        area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if area < best:
            best, ba = area, a
    return float(ba)

def pick_strategy(rb, meta, imgs, C, T):
    a = cube_yaw(meta, imgs)
    # finger axis at yaw psi is psi+90deg; cube faces at a, a+90: choose psi = a (mod 90) nearest 0
    ys = [wrap(a + k * np.pi / 2) for k in range(4)]
    y = min(ys, key=lambda v: abs(v))
    log("pick yaw", np.degrees(y))
    zc = max(0.012, C[2] - 0.022)
    move_to(rb, [C[0], C[1], 0.12], yaw=y, grip=-1, vmax=0.02)
    meta, imgs, C2, _ = scene(rb)
    if C2 is not None and np.linalg.norm(C2[:2] - C[:2]) < 0.04:
        C = C2
        move_to(rb, [C[0], C[1], 0.12], yaw=y, grip=-1, vmax=0.01)
    move_to(rb, [C[0], C[1], zc], yaw=y, grip=-1, vmax=0.012)
    hold(rb, 12, 1.0)
    log("pick width", rb.grip_open())
    move_to(rb, [C[0], C[1], 0.12], yaw=y, grip=1, vmax=0.012)
    move_to(rb, [T[0], T[1], 0.12], yaw=y, grip=1, vmax=0.015)
    move_to(rb, [T[0], T[1], zc + 0.01], yaw=y, grip=1, vmax=0.01)
    hold(rb, 10, -1.0)
    p = rb.pos()
    move_to(rb, [p[0], p[1], 0.15], yaw=y, grip=-1, vmax=0.012)
    return True

# ---------------------------------------------------------------- push
def push_strategy(rb, meta, imgs, C, T, grip=1.0):
    for attempt in range(4):
        D = T[:2] - C[:2]; dist = float(np.linalg.norm(D))
        log("push attempt", attempt, "dist", dist)
        if dist < 0.008:
            break
        d = D / dist
        y = wrap(np.arctan2(d[1], d[0]))  # finger axis (y+90deg) perpendicular to d
        y = min([y, wrap(y + np.pi)], key=lambda v: abs(v))
        gap = 0.03
        back = 0.02 + 0.012 + gap
        p0 = C[:2] - d * back
        seg = min(dist, 0.12) if attempt < 3 else dist
        p1 = p0 + d * (gap + seg)
        move_to(rb, [p0[0], p0[1], 0.12], yaw=y, grip=grip, vmax=0.02)
        move_to(rb, [p0[0], p0[1], 0.02], yaw=y, grip=grip, vmax=0.012)
        move_to(rb, [p1[0], p1[1], 0.02], yaw=y, grip=grip, vmax=0.006, tol=0.003)
        if rb.ended: break
        p = rb.pos()
        move_to(rb, [p[0] - d[0] * 0.02, p[1] - d[1] * 0.02, 0.02], yaw=y, grip=grip, vmax=0.006, max_steps=10)
        move_to(rb, [p[0], p[1], 0.15], yaw=y, grip=grip, vmax=0.015)
        if rb.ended: break
        meta, imgs, C2, T2 = scene(rb)
        if C2 is None: break
        C = C2
        if T2 is not None and np.linalg.norm(T2[:2] - T[:2]) < 0.03:
            T = T2
    return True

# ---------------------------------------------------------------- main
def main(ctx):
    try:
        OUT.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    t0 = time.time()
    rb = M.Robot(ctx)
    summary = {}
    try:
        r = ctx.call("robomme_observe")
        meta = M._meta(r)
        rb._upd(meta)
        imgs = M.decode_images(r, meta)
        seq = meta.get("frame_sequence") or {}
        man = seq.get("manifest_path") or os.environ.get("MOVECUBE_MANIFEST")
        vinfo = {"manner": "tool", "tool_kind": "hook"}
        if man:
            try:
                vinfo = analyze(man)
            except Exception as e:
                log("video analysis failed", repr(e))
        log("video", json.dumps(vinfo))
        summary["video"] = vinfo
        c = P.locate_cube(meta, imgs)
        T, _ = P.locate_target(meta, imgs)
        C = None if c is None else c[0]
        log("initial cube", C, "target", T)
        if C is None or T is None:
            go_home_view(rb, -1)
            meta, imgs, C, T = scene(rb)
        if C is None or T is None:
            goto(rb, [-0.06, 0.0, 0.45], 0.0, -1, vmax=0.03)
            meta, imgs, C, T = scene(rb)
        if C is None or T is None:
            log("cannot find cube/target"); return
        manner = vinfo.get("manner", "hook")
        if manner in ("tool", "hook"):
            kind = vinfo.get("tool_kind") or "hook"
            ok = tool_strategy(rb, meta, imgs, C, T, kind)
            if not ok and not rb.ended:
                log("tool strategy failed; retry once")
                meta, imgs, C, T = scene(rb)
                if C is not None and T is not None:
                    tool_strategy(rb, meta, imgs, C, T, kind)
        elif manner == "ramp":
            ramp_strategy(rb, meta, imgs, C, T, vinfo)
        elif manner == "pick":
            pick_strategy(rb, meta, imgs, C, T)
        else:
            grip = 1.0 if vinfo.get("width_min", 0.08) < 0.07 else -1.0
            push_strategy(rb, meta, imgs, C, T, grip=grip)
        if not rb.ended:
            hold(rb, 10, rb.state[6] < 0.03 and 1.0 or -1.0)
    except Exception:
        log("EXC", traceback.format_exc())
    finally:
        summary.update({"ended": rb.ended, "complete": rb.complete, "step": rb.step_id, "time": time.time() - t0})
        log("summary", json.dumps(summary, default=str))
        try:
            (OUT / "controller_log.txt").write_text("\n".join(M.LOG))
        except Exception:
            pass
