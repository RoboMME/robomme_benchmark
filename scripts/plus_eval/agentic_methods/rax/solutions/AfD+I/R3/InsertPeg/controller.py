"""InsertPeg controller v2 (RAX, RoboMME-Plus).

Strategy:
 * The conditioning video is recorded in the same scene and carries the expert's EE state per frame.
   The expert's path is IK-feasible, so approach (start -> hover above the peg) and transport
   (grasp -> above the pre-insert point) follow the video states as waypoints. Replaying the grasp
   pose grasps the same peg at the same end. The final straight segment gives the insertion
   direction u and the insertion orientation.
 * The box hole position differs between video and execution (start-1 finding), so the hole is located
   in the current scene before grasping: the empty gripper, in the expert's insertion orientation
   (no tilt -> known-feasible wrist pose), is placed so that the wrist camera sees the entry face
   ~38 deg off-axis. Depth pixels deeper than the face plane are traced back to it -> opening edges;
   the hole floor gives its bottom.
 * The peg axis line is aligned to the hole centre and pushed along u with stall detection and a
   small spiral search.
 * Native IK errors cost 100 steps and do not move the arm (start-2 finding): every step call checks
   the step clock; a failing motion is abandoned instead of repeated.
"""
import json
import os
import time
import traceback

import numpy as np

from motion import Robot, rot_err, rotvec_to_mat, mat_to_rotvec
from percep import images_from_reply, box_from_front, detect_hole, peg_line, load_depth_png
from plan import analyze_video, frame_uw, uwz_to_xyz, view_pose_offaxis

OUT = "/workspace/output"
T0 = time.time()
LOGF = None

PEG_Z_OFF = -0.0006      # peg centre z relative to EE after the replayed grasp (start-1 measurement)
SEARCH = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, 2), (2, -2), (-2, -2),
          (4, 0), (-4, 0), (0, 4), (0, -4), (4, 3), (-4, 3), (4, -3), (-4, -3), (6, 0), (-6, 0), (0, 6), (0, -6)]
MAX_IK = 3               # stop issuing new risky motions after this many IK penalties


def log(*a):
    global LOGF
    msg = "[%.1fs] " % (time.time() - T0) + " ".join(str(x) for x in a)
    print(msg, flush=True)
    try:
        if LOGF is None:
            os.makedirs(OUT, exist_ok=True)
            LOGF = open(os.path.join(OUT, "controller_log.txt"), "a")
        LOGF.write(msg + "\n"); LOGF.flush()
    except Exception:
        pass


def load_video(meta):
    seq = meta.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if not mp:
        raise RuntimeError("no frame_sequence manifest in the initial observation")
    path = mp if os.path.isabs(mp) else os.path.join("/workspace", mp)
    man = json.load(open(path))
    frames = sorted(man["frames"], key=lambda f: f["frame_index"])
    vid = [f for f in frames if f.get("phase") == "video_demo"]
    return vid, frames


def observe(r, ctx):
    rep = ctx.call("robomme_observe")
    meta, ims = images_from_reply(rep)
    r._update(meta)
    return meta, ims


def budget_ok(r, need=10):
    return (not r.ended) and (r.remaining is None or r.remaining > need)


def follow(r, S, idxs, grip, tol_p=0.012, tol_r=0.10, final_tol=(0.003, 0.02), max_steps=200,
           max_lin=0.025, max_ang=0.2, final_max_lin=0.01):
    """Track video states S[idxs] as waypoints (carrot following). Returns True when the last one is reached."""
    idxs = list(idxs)
    if not idxs:
        return True
    k = 0
    n = 0
    while n < max_steps and budget_ok(r, 2):
        while k < len(idxs) - 1:
            t = S[idxs[k]]
            if np.linalg.norm(t[:3] - r.pos) < tol_p and np.linalg.norm(rot_err(t[3:6], r.rot)) < tol_r:
                k += 1
            else:
                break
        t = S[idxs[k]]
        last = k == len(idxs) - 1
        ep = np.linalg.norm(t[:3] - r.pos); er = np.linalg.norm(rot_err(t[3:6], r.rot))
        if last and ep < final_tol[0] and er < final_tol[1]:
            return True
        r.step([r.row_toward(t[:3], t[3:6], grip, max_lin=(final_max_lin if last else max_lin), max_ang=max_ang, gain=1.5)])
        n += 1
        if r.ik_flag:
            log("follow: IK penalty at waypoint %d/%d (frame %d)" % (k, len(idxs), idxs[k]))
            return False
    log("follow: max steps / budget, waypoint %d/%d" % (k, len(idxs)))
    return False


def inspect_hole(r, ctx, plan, box, u, R_ins, z_travel):
    """Move the empty gripper (insertion orientation) so the wrist camera sees the entry face off-axis."""
    uu, ww = frame_uw(u)
    wc = 0.5 * (box["w_lo"] + box["w_hi"])
    F = uwz_to_xyz(box["a_face"], wc, 0.5 * box["z_top"], u)
    # rotate to the insertion orientation at travel height (the expert does this rotation at ~lift height)
    up = r.pos.copy(); up[2] = max(up[2], z_travel)
    ok, n, ep, er = r.goto(up, R_ins, -1, tol_p=0.01, tol_r=0.03, max_steps=60, max_lin=0.025, max_ang=0.2, gain=1.5)
    if r.ik_flag:
        return None
    for beta, dist in ((38, 0.16), (31, 0.18)):
        p, rv, C, pix = view_pose_offaxis(F, u, R_ins, np.radians(beta), dist)
        log("view beta %d: EE target %s, F expected pixel %s" % (beta, np.round(p, 4), np.round(pix, 1)))
        if p[2] < 0.06 or not (8 < pix[0] < 248 and 8 < pix[1] < 248):
            log("view pose too low or face outside the image, skip")
            continue
        if r.ik_errors >= 2 or not budget_ok(r, 300):
            log("inspection stopped: ik errors %d remaining %s" % (r.ik_errors, r.remaining))
            return None
        above = p.copy(); above[2] = max(p[2], z_travel)
        r.goto(above, R_ins, -1, tol_p=0.01, tol_r=0.03, max_steps=60, max_lin=0.025, max_ang=0.2, gain=1.5)
        if r.ik_flag:
            return None
        ok, n, ep, er = r.goto(p, R_ins, -1, tol_p=0.003, tol_r=0.015, max_steps=50, max_lin=0.02, max_ang=0.2, gain=1.5)
        log("view reached=%s steps=%d ep=%.4f er=%.4f pos=%s" % (ok, n, ep, er, np.round(r.pos, 4)))
        if r.ik_flag:
            # back off upward along the descent column and try the next (higher) view
            r.goto(above, R_ins, -1, tol_p=0.01, tol_r=0.03, max_steps=30, max_lin=0.02, gain=1.5)
            continue
        r.hold(-1, 2)
        meta, ims = observe(r, ctx)
        dep = ims.get(("wrist", "depth"))
        if dep is None:
            continue
        try:
            np.save(os.path.join(OUT, "view_wrist_depth_b%d.npy" % beta), dep)
        except Exception:
            pass
        h = detect_hole(dep, meta["camera_intrinsics"]["wrist"]["matrix"], meta["camera_poses"]["wrist"], u,
                        a_face=box["a_face"], w_range=(box["w_lo"], box["w_hi"]), z_top=box["z_top"], log=log)
        if h is not None:
            log("hole: w %.4f z %.4f (z_open %.4f floor %.4f) n %d span w %.4f z %.4f (w %.4f..%.4f z %.4f..%.4f) a_face %.4f" % (
                h["w"], h["z"], h["z_open"], h["z_floor"], h["n"], h["w_span"], h["z_span"], h["w_lo"], h["w_hi"],
                h["z_lo"], h["z_hi"], h["a_face"]))
            if 0.012 < h["w_span"] < 0.045 and 0.008 < h["z_span"] < 0.045 and h["n"] >= 30:
                return h
            log("hole span implausible, try next view")
        else:
            log("no hole found at beta", beta)
    return None


def do_grasp(r, S, plan, hover_idx):
    gi = plan["grasp_idx"]
    g = S[gi]
    gp, gr = g[:3], g[3:6]
    ok = follow(r, S, list(range(hover_idx, gi, 3)) + [gi], -1, tol_p=0.008, tol_r=0.06, final_tol=(0.0008, 0.006),
                max_steps=120, max_lin=0.015, final_max_lin=0.006)
    log("at grasp: ok=%s pos=%s video=%s rot err %.4f" % (ok, np.round(r.pos, 4), np.round(gp, 4),
                                                           np.linalg.norm(rot_err(gr, r.rot))))
    if r.ik_flag:
        return False
    for i in range(14):
        r.step([r.row_toward(gp, gr, 1, max_lin=0.004)])
        if r.ik_flag or r.ended:
            break
    log("closed width %.4f (video %.4f)" % (r.width, plan["closed_width"]))
    return r.width > 0.006


def push(r, u, wk, zk, R_ins, a_limit, a_fast, lead_slow=0.0012, max_n=160):
    """Push along u on the line (wk, zk) until stall / a_limit. Returns final a."""
    uu, ww = frame_uw(u)
    hist = []
    while budget_ok(r, 3) and not r.complete:
        a_now = float(r.pos[:2] @ uu)
        lead = 0.003 if a_now < a_fast else lead_slow
        tgt = uwz_to_xyz(min(a_now + lead, a_limit), wk, zk, u)
        r.step([r.row_toward(tgt, R_ins, 1, max_lin=0.004, gain=1.5)])
        if r.ik_flag:
            break
        a_now = float(r.pos[:2] @ uu)
        hist.append(a_now)
        if len(hist) > 7 and hist[-1] - hist[-7] < 0.0006:
            break
        if a_now >= a_limit - 0.0005 or len(hist) > max_n:
            break
    return float(r.pos[:2] @ uu)


def insert(r, ctx, plan, hole, u, box):
    uu, ww = frame_uw(u)
    R_ins = plan["end"][3:6]
    a_end = float(plan["end"][:2] @ uu)
    a_face = hole["a_face"] if hole is not None else (box["a_face"] if box is not None else None)
    a_pre = float(plan["pre"][:2] @ uu)
    if a_face is not None:
        a_pre = min(a_pre, a_face - 0.13)          # tip (~8.3 cm ahead of EE) well before the face
        a_fast = a_face - 0.10                      # fast approach until ~1.7 cm before expected contact
    else:
        a_fast = a_pre + 0.4 * (a_end - a_pre)
    a_in = a_end - 0.028                            # stops beyond this are inside the hole
    if hole is not None:
        w_t, z_t = hole["w"], hole["z"] - PEG_Z_OFF
    else:
        w_t, z_t = float(plan["end"][:2] @ ww), float(plan["end"][2])
    w_v, z_v = float(plan["end"][:2] @ ww), float(plan["end"][2])
    log("insert line w %.4f z %.4f (video w %.4f z %.4f; shift %.1f, %.1f mm) a_pre %.4f a_fast %.4f a_in %.4f a_end %.4f" % (
        w_t, z_t, w_v, z_v, (w_t - w_v) * 1000, (z_t - z_v) * 1000, a_pre, a_fast, a_in, a_end))
    # above the pre-insert point at the current (transport) height, then descend
    z_hi = max(r.pos[2], z_t + 0.03)
    r.goto(uwz_to_xyz(a_pre, w_t, z_hi, u), R_ins, 1, tol_p=0.003, tol_r=0.01, max_steps=60, max_lin=0.02, gain=1.5)
    pre = uwz_to_xyz(a_pre, w_t, z_t, u)
    r.goto(pre + np.array([0, 0, 0.012]), R_ins, 1, tol_p=0.002, tol_r=0.008, max_steps=50, max_lin=0.015, gain=1.5)
    ok, n, ep, er = r.goto(pre, R_ins, 1, tol_p=0.0007, tol_r=0.005, max_steps=40, max_lin=0.006, gain=1.0)
    log("pre-insert ok=%s ep=%.4f er=%.4f pos=%s width %.4f step %s" % (ok, ep, er, np.round(r.pos, 4), r.width, r.step_id))
    try:
        meta, ims = observe(r, ctx)
        L = peg_line(ims[("wrist", "depth")], meta["camera_intrinsics"]["wrist"]["matrix"], meta["camera_poses"]["wrist"], r.pos, uu)
        if L:
            log("peg line: ps %.4f po %.5f zs %.4f zo %.5f n %d amax %.3f" % (L["ps"], L["po"], L["zs"], L["zo"], L["n"], L["amax"]))
    except Exception as e:
        log("peg line failed", e)
    a_contact = None
    for k, (dw, dz) in enumerate(SEARCH):
        if r.ended or r.complete or not budget_ok(r, 25) or r.ik_errors >= MAX_IK:
            break
        wk, zk = w_t + dw * 0.001, z_t + dz * 0.001
        a_start = a_pre if a_contact is None else a_contact - 0.006
        r.goto(uwz_to_xyz(a_start, wk, zk, u), R_ins, 1, tol_p=0.0006, tol_r=0.004, max_steps=40, max_lin=0.005, gain=1.2)
        a0 = float(r.pos[:2] @ uu)
        a_now = push(r, u, wk, zk, R_ins, a_end + 0.006, a_fast if a_contact is None else -1e9)
        log("probe %d off(%d,%d)mm: a %.4f -> %.4f (in>%.4f end %.4f) step %s complete %s width %.4f" % (
            k, dw, dz, a0, a_now, a_in, a_end, r.step_id, r.complete, r.width))
        if r.complete or r.ended:
            break
        if a_now > a_in:
            # inside the hole: keep pushing with small wiggles toward the video's final depth
            log("peg inside hole (%.1f mm before video end)" % ((a_end - a_now) * 1000))
            for wig in ((0, 0), (0.001, 0), (-0.001, 0), (0, 0.001), (0, -0.001), (0, 0), (0.0015, 0.0015), (-0.0015, -0.0015)):
                if r.complete or not budget_ok(r, 3):
                    break
                a_now = push(r, u, wk + wig[0], zk + wig[1], R_ins, a_end + 0.02, -1e9, lead_slow=0.0015, max_n=40)
                log("  wiggle %s -> a %.4f complete %s" % (wig, a_now, r.complete))
            if not r.complete and budget_ok(r, 10):
                for _ in range(6):
                    r.step([r.row_toward(r.pos, R_ins, 1, max_lin=0.002)])
                    if r.complete or r.ended:
                        break
            break
        a_contact = a_now if a_contact is None else min(a_contact, a_now)
        r.goto(uwz_to_xyz(a_contact - 0.006, wk, zk, u), R_ins, 1, tol_p=0.0008, tol_r=0.005, max_steps=25, max_lin=0.004)
    return r.complete


def main(ctx):
    r = Robot(ctx, log=log)
    try:
        rep = ctx.call("robomme_observe")
        meta, ims = images_from_reply(rep)
        r._update(meta)
        log("start step", r.step_id, "remaining", r.remaining, "state", np.round(r.state, 4).tolist())
        vid, frames = load_video(meta)
        S = np.array([f["state"] for f in vid], float)
        plan = analyze_video(S)
        u = plan["u"]
        uu, ww = frame_uw(u)
        R_ins = plan["end"][3:6]
        log("video plan:", {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v) for k, v in plan.items()})
        gi, ci, pi = plan["grasp_idx"], plan["close_idx"], plan["pre_idx"]
        gz = S[gi, 2]
        hov_c = [k for k in range(gi) if S[k, 2] > gz + 0.07]
        hover_idx = max(hov_c) if hov_c else 0
        desc_c = [k for k in range(ci, pi) if S[k, 2] > S[pi, 2] + 0.03]
        desc_idx = max(desc_c) if desc_c else ci
        z_travel = float(min(plan["lift_z"], 0.24))
        log("hover idx %d %s, desc idx %d %s, travel z %.3f" % (hover_idx, np.round(S[hover_idx, :3], 3), desc_idx,
                                                              np.round(S[desc_idx, :3], 3), z_travel))
        try:
            f0 = vid[0]
            files = {(x["camera"], x["kind"]): x["path"] for x in f0["files"]}
            d0 = load_depth_png(os.path.join("/workspace", files[("front", "depth")]))
            diff = np.abs(d0 - ims[("front", "depth")])
            log("scene check: front depth pixels differing >3mm from video frame 0: %d" % int((diff > 0.003).sum()))
        except Exception as e:
            log("scene check failed", e)
        box = box_from_front(ims[("front", "depth")], meta["camera_intrinsics"]["front"]["matrix"],
                             meta["camera_poses"]["front"], uu, plan["end"][:2] + 0.07 * uu, radius=0.15, log=log)
        # 1) approach along the expert's path to the hover point above the peg
        ok = follow(r, S, list(range(0, hover_idx, 4)) + [hover_idx], -1, max_steps=150)
        log("hover reached=%s pos %s step %s" % (ok, np.round(r.pos, 4), r.step_id))
        # 2) locate the hole with the empty gripper
        hole = None
        if box is not None and r.ik_errors == 0 and budget_ok(r, 300):
            try:
                hole = inspect_hole(r, ctx, plan, box, u, R_ins, z_travel)
            except Exception as e:
                log("inspect failed", e, traceback.format_exc())
        if hole is not None:
            dv = np.array([hole["w"] - float(plan["end"][:2] @ ww), hole["z"] - PEG_Z_OFF - plan["end"][2]])
            if np.linalg.norm(dv) > 0.035:
                log("hole %.1f mm from the video line: rejected" % (np.linalg.norm(dv) * 1000))
                hole = None
        if hole is None:
            log("FALLBACK: hole not localized, using the video insertion line")
        if r.ended:
            return
        # 3) back above the hover point and down along the expert's grasp approach
        hv = S[hover_idx]
        p_up = hv[:3].copy(); p_up[2] = max(z_travel, hv[2])
        if np.linalg.norm(r.pos[:2] - hv[:2]) > 0.02:
            up = r.pos.copy(); up[2] = max(up[2], p_up[2])
            r.goto(up, r.rot, -1, tol_p=0.01, tol_r=0.05, max_steps=40, max_lin=0.025, gain=1.5)
            r.goto(p_up, hv[3:6], -1, tol_p=0.008, tol_r=0.03, max_steps=80, max_lin=0.025, max_ang=0.2, gain=1.5)
        r.goto(hv[:3], hv[3:6], -1, tol_p=0.004, tol_r=0.02, max_steps=60, max_lin=0.02, max_ang=0.2, gain=1.5)
        got = do_grasp(r, S, plan, hover_idx)
        if not got and budget_ok(r, 200) and not r.ik_flag:
            log("grasp seems empty (width %.4f): retry once" % r.width)
            r.hold(-1, 6)
            r.goto(hv[:3], hv[3:6], -1, tol_p=0.004, tol_r=0.02, max_steps=40, max_lin=0.02, gain=1.5)
            got = do_grasp(r, S, plan, hover_idx)
        # 4) transport along the expert's path (lift, rotate, carry) to above the pre-insert point
        ok = follow(r, S, list(range(ci + 1, desc_idx, 4)) + [desc_idx], 1, max_steps=200)
        log("transport reached=%s pos %s width %.4f step %s" % (ok, np.round(r.pos, 4), r.width, r.step_id))
        if r.ik_flag and budget_ok(r, 100):
            # recover: lift straight up and go directly
            up = r.pos.copy(); up[2] = max(up[2], z_travel)
            r.goto(up, r.rot, 1, tol_p=0.01, tol_r=0.05, max_steps=30, max_lin=0.02, gain=1.5)
        # 5) align with the hole and insert
        if budget_ok(r, 30):
            insert(r, ctx, plan, hole, u, box)
        log("final: complete=%s ended=%s step=%s remaining=%s pos=%s ik_errors=%d" % (
            r.complete, r.ended, r.step_id, r.remaining, np.round(r.pos, 4), r.ik_errors))
    except Exception as e:
        log("CONTROLLER ERROR", e, traceback.format_exc())
    finally:
        log("exit; calls", r.calls)
