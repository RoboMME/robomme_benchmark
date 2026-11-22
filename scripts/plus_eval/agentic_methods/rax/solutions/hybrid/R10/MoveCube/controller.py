"""MoveCube controller: imitate the manner in the conditioning video.
Detect the grasped object in the video (robot state + depth), match it in the execution scene,
grasp it and push the cube onto the target with its far end."""
import json, math, time, os, sys, traceback
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from motion import Robot, packet, down_yaw, aa_to_mat, mat_to_aa, rot_error
from percep import *
from plan import *

OUT = Path("/workspace/output")
LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "controller_log.txt", "a") as f:
            f.write(s + "\n")
    except Exception:
        pass

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def cur_yaw(state):
    R = aa_to_mat(state[3:6]); gx = R[:, 0]
    return math.atan2(gx[1], gx[0])

def latest_exec_frame(p):
    """from a robomme_observe/step packet, return frame dict usable with load_frame."""
    seq = p["frame_sequence"]
    man = json.loads((WS / seq["manifest_path"]).read_text())
    return man["frames"][-1], man

def decode_reply(rep):
    import base64, io
    from PIL import Image as _I
    blocks = rep["content"] if isinstance(rep, dict) else rep
    p = packet(blocks)
    out = {}
    for a in p.get("attachments", []):
        b = blocks[a["content_index"]]
        im = _I.open(io.BytesIO(base64.b64decode(b["data"])))
        out[(a["camera"], a["kind"])] = np.array(im)
    return p, out

def scene_now(robot):
    rep = robot.ctx.call("robomme_observe")
    p, ims = decode_reply(rep)
    robot._upd(p)
    rgb = ims[("front", "rgb")][..., :3].astype(np.int16)
    dep = ims[("front", "depth")].astype(np.float32) * 0.001
    pw = backproject(dep, p["camera_intrinsics"]["front"], p["camera_poses"]["front"])
    robot.last_dep = dep
    return p, rgb, pw

def measure_tip(robot):
    """held tool tip (farthest tool point from TCP) seen by the front camera; robot must be lifted (z>=0.15)."""
    p, rgb, pw = scene_now(robot)
    tcp = robot.pos()
    z = pw[..., 2]
    wm = wood_mask(rgb); cc = color_class(rgb)
    dxy = np.linalg.norm(pw[..., :2] - tcp[:2], axis=-1)
    m = (np.isfinite(z) & (z > 0.065) & (z < tcp[2] + 0.015) & (~wm) & (cc != "blue") & (cc != "red")
         & (dxy < 0.45) & (dxy > 0.05))
    # drop points near the arm (white, close to the TCP in xy & high)
    if m.sum() < 5:
        log("measure_tip: no points"); return None, None
    pts = pw[m]; d = np.linalg.norm(pts[:, :2] - tcp[:2], axis=1)
    # require continuity: sort by distance and cut at first gap > 6 cm
    order = np.argsort(d); ds = d[order]
    gaps = np.where(np.diff(ds) > 0.06)[0]
    cut = gaps[0] + 1 if len(gaps) else len(ds)
    keep = pts[order[:cut]]; dk = ds[:cut]
    if len(keep) < 5:
        log("measure_tip: too few"); return None, None
    sel = keep[dk > dk.max() - 0.02]
    tip = sel.mean(0)
    log("measure_tip n", int(m.sum()), "kept", len(keep), "tip", np.round(tip, 3).tolist(), "len", round(float(np.linalg.norm(tip[:2] - tcp[:2])), 3))
    return tip[:2], tip[2]

DEP0 = {}
ALL_BLOBS = []
PRE_GRIP = -1.0

def find_tip_near(robot, expect_xy, radius=0.10):
    """locate the low end of the held tool near expect_xy via depth change vs the initial scene."""
    p, rgb, pw = scene_now(robot)
    dep = robot.last_dep; dep0 = DEP0.get("front")
    z = pw[..., 2]
    cc = color_class(rgb)
    changed = np.abs(dep - dep0) > 0.008 if dep0 is not None else np.ones_like(dep, bool)
    dxy = np.linalg.norm(pw[..., :2] - np.asarray(expect_xy)[None, None, :], axis=-1)
    m = changed & np.isfinite(z) & (z > -0.005) & (z < 0.09) & (cc != "red") & (dxy < radius)
    if m.sum() < 4:
        log("find_tip_near: none (n=%d)" % int(m.sum())); return None, None
    pts = pw[m]
    zmin = np.percentile(pts[:, 2], 5)
    sel = pts[pts[:, 2] < zmin + 0.02]
    tip = sel[:, :2].mean(0)
    log("find_tip_near n", int(m.sum()), "tip", np.round(tip, 3).tolist(), "zmin", round(float(zmin), 3), "err", round(float(np.linalg.norm(tip - expect_xy)), 3))
    return tip, float(zmin)

def rotate_to_yaw(robot, yaw, max_steps=80, gain=0.8):
    tgt = down_yaw(yaw)
    used = 0
    while used < max_steps and not robot.ended:
        e = rot_error(robot.state[3:6], tgt)
        if np.linalg.norm(e) < 0.03:
            return True
        st = np.clip(e / 0.5, -0.5, 0.5) * gain
        robot.step([[0, 0, 0, st[0], st[1], st[2], robot.gripper]] * 3)
        used += 3
    return False

def choose_grasp_yaw(ax, cur):
    a = math.atan2(ax[1], ax[0])
    cands = [wrap(a), wrap(a + math.pi)]
    return min(cands, key=lambda y: abs(wrap(y - cur)))

BASE = np.array([-0.615, 0.0])

def choose_push_rotation(off, d, c, max_dev=math.radians(70), obstacles=None):
    """rotation phi (about z) applied to tool offset off (TCP->tip).
    The handle direction h=-off_rot must stay within max_dev of -d (shaft behind the tip),
    and should point toward the robot base so the TCP stays reachable."""
    L = np.linalg.norm(off)
    a_off = math.atan2(off[1], off[0])
    best = None
    for k in range(-36, 37):
        phi = math.radians(5 * k)
        if abs(phi) > 2.6:
            continue
        o = np.array([math.cos(a_off + phi), math.sin(a_off + phi)]) * L
        h = -o / L
        dev = math.acos(max(-1, min(1, float(np.dot(h, -d)))))
        if dev > max_dev:
            continue
        tcp_mid = c - o - d * 0.05
        r = np.linalg.norm(tcp_mid - BASE)
        cost = abs(r - 0.50) * 4 + abs(phi) * 0.15 + dev * 0.3
        if obstacles is not None and len(obstacles):
            for q in (c - o - d * 0.05, c - o * 0.5 - d * 0.05, c - o, c - o * 0.5):
                dm = np.min(np.linalg.norm(obstacles - q, axis=1))
                if dm < 0.07:
                    cost += (0.07 - dm) * 40
        if best is None or cost < best[0]:
            best = (cost, phi)
    return best[1] if best else 0.0

def _main(ctx):
    t0 = time.time()
    robot = Robot(ctx, log)
    p = robot.observe()
    log("start state", np.round(robot.state, 3).tolist(), "limit", p.get("episode_limits"))
    if "frame_sequence" in p:
        man = json.loads((WS / p["frame_sequence"]["manifest_path"]).read_text())
    else:
        cands = sorted((WS / "runtime/autonomous_controller/runs").glob("*/frames/sequence_000000/manifest.json"),
                       key=lambda q: q.stat().st_mtime)
        log("no frame_sequence in observe; fallback manifest", cands[-1] if cands else None)
        man = json.loads(cands[-1].read_text())
    frames = man["frames"]
    vinfo = analyze_video(frames, log)
    log("video", {k: v for k, v in (vinfo or {}).items() if not k.startswith("_")})
    if "frame_sequence" in p:
        f = [x for x in frames if x["phase"] == "execution"][-1]
        rgb, dep = load_frame(f)
        intr = f.get("camera_intrinsics") or p["camera_intrinsics"]; pose = f.get("camera_poses") or p["camera_poses"]
    else:
        # mid-episode start: open, lift clear and look at the current scene
        robot.set_gripper(-1.0, 10)
        q = robot.pos(); robot.move_to([q[0], q[1], 0.35], tol=0.02, max_steps=80)
        robot.move_to([-0.06, 0.0, 0.40], tol=0.03, max_steps=120, rot_target=down_yaw(0.0))
        p, rgb, _pw = scene_now(robot)
        dep = robot.last_dep
        intr = p["camera_intrinsics"]; pose = p["camera_poses"]
    DEP0["front"] = dep
    blobs, lab, pw = scene_blobs(rgb, dep, intr["front"], pose["front"])
    blobs = [b for b in blobs if not is_robot(b)]
    ALL_BLOBS[:] = blobs
    cube, target = classify_static(blobs)
    if cube is None or target is None:
        log("cube/target not found"); return
    c = cube["center"]; t = target["center"]
    cc0 = color_center(rgb, pw, "red", zmin=0.005)
    if cc0 is not None:
        c = cc0; cube["center"] = cc0
    log("cube", np.round(c, 3).tolist(), "target", np.round(t, 3).tolist())
    manner = (vinfo or {}).get("manner", "unknown")
    if manner in ("tool", "ball"):
        cands = match_tool(vinfo, blobs, cube, target)
        for s, b in cands[:4]:
            log("cand", round(s, 2), describe(b))
        if not cands:
            log("no tool candidates"); return
        tool = cands[0][1]
        ramp = find_ramp(blobs)
        if manner == "ball" and ramp is not None:
            ball_ramp(robot, tool, ramp, rgb, pw, c, t)
            log("done; ended", robot.ended, "complete", robot.complete, "steps", robot.step_id)
            return
        obst = obstacles_from(blobs, exclude=[tool, cube, target])
        tool_push(robot, tool, vinfo, rgb, pw, cube, target, manner, obst)
    else:
        log("manner not implemented:", manner)
    log("done; ended", robot.ended, "complete", robot.complete, "steps", robot.step_id, "wall", round(time.time() - t0, 1))

def find_ramp(blobs):
    best = None
    for b in blobs:
        h = dict(zip(CLASSES, b["hist"]))
        if h["blue"] > 0.5 and b["n"] > 800 and (best is None or b["n"] > best["n"]):
            best = b
    return best

def ball_ramp(robot, ball, ramp, rgb, pw, c, t):
    """grasp the ball and release it on the ramp's high side, in line with the cube."""
    pts = ramp["pts"]; pts = pts[np.isfinite(pts[:, 2])]
    top = pts[pts[:, 2] > 0.01]
    rng = np.random.default_rng(0)
    bestp = None
    for _ in range(300):
        i = rng.choice(len(top), 3, replace=False)
        P = top[i]; nrm = np.cross(P[1] - P[0], P[2] - P[0])
        if np.linalg.norm(nrm) < 1e-9:
            continue
        nrm = nrm / np.linalg.norm(nrm)
        if abs(nrm[2]) < 0.6:          # sloped face, not a vertical side
            continue
        inl = np.abs((top - P[0]) @ nrm) < 0.004
        if bestp is None or inl.sum() > bestp[0]:
            bestp = (inl.sum(), nrm, inl)
    _, nrm, inl = bestp
    face = top[inl]
    A = np.c_[face[:, 0], face[:, 1], np.ones(len(face))]
    coef, *_ = np.linalg.lstsq(A, face[:, 2], rcond=None)
    g = coef[:2]; up = g / (np.linalg.norm(g) + 1e-9); down = -up
    zmax = float(np.percentile(face[:, 2], 98))
    log("ramp face pts", len(face), "slope", np.round(coef, 3).tolist(), "up dir", np.round(up, 3).tolist(), "zmax", round(zmax, 3))
    # ball path: straight downhill line through the cube centre (small bias away from target side)
    perp = np.array([-down[1], down[0]])
    lat = float(np.dot(t - c, perp))
    bias = 0.0
    base_pt = c + perp * bias
    face_lat = (face[:, :2] - base_pt) @ perp
    face_s = (face[:, :2] - base_pt) @ up
    on = np.abs(face_lat) < 0.01
    if on.sum() < 5:
        log("cube line misses ramp face"); return
    # lateral margin to the face edges
    lat_all = (face[:, :2] - c) @ perp
    log("ramp lateral extent", round(float(lat_all.min()), 3), round(float(lat_all.max()), 3), "bias", round(bias, 3))
    smax = float(np.percentile(face_s[on], 95)); smin = float(np.percentile(face_s[on], 5))
    # release height as in the demonstrated video: surface z ~0.034 for a 0.177 m cube-target distance
    # calibrated in exploration 9: effective drop (surface z + 3 mm gap) 0.043 m pushed the cube 0.303 m;
    # cube travel ~ proportional to drop height, so drop = 0.145 * needed travel
    h_rel = float(np.clip(0.145 * np.linalg.norm(t - c) - 0.003, 0.015, 0.07))
    slope = float(np.linalg.norm(coef[:2]))
    z_at = lambda q_: float(coef[0] * q_[0] + coef[1] * q_[1] + coef[2])
    s_rel = smin + (h_rel - z_at(base_pt + up * smin)) / max(slope, 0.05)
    s_rel = float(np.clip(s_rel, smin + 0.01, smax - 0.01))
    q = base_pt + up * s_rel
    zq = z_at(q)
    log("release height target", round(h_rel, 3))
    best = (q, zq)
    q, zq = best
    log("release point", np.round(q, 3).tolist(), "surface z", round(zq, 3))
    # grasp ball
    bp = ball["pts"]; bc = np.median(bp[:, :2], axis=0); btop = float(np.nanmax(bp[:, 2]))
    gz = max(btop * 0.5, 0.015)
    robot.gripper = -1.0
    robot.rot_hold = down_yaw(0.0)
    robot.move_to([bc[0], bc[1], 0.15], speed=0.8, tol=0.01)
    robot.move_to([bc[0], bc[1], gz + 0.05], speed=0.5, tol=0.005)
    robot.move_to([bc[0], bc[1], gz], speed=0.2, tol=0.003)
    robot.set_gripper(1.0, 15)
    w = robot.state[6] + robot.state[7]
    log("ball grasp width", round(w, 4))
    robot.move_to([bc[0], bc[1], 0.28], speed=0.4, tol=0.01)
    robot.move_to([q[0], q[1], 0.28], speed=0.5, tol=0.008)
    rz = zq + btop * 0.5 + 0.003
    robot.move_to([q[0], q[1], rz], speed=0.2, tol=0.004, max_steps=80)
    robot.set_gripper(-1.0, 10)
    robot.move_to([q[0], q[1], 0.30], speed=0.4, tol=0.01, max_steps=40)
    robot.hold(60)
    p, rgb2, pw2 = scene_now(robot)
    cc2 = color_center(rgb2, pw2, "red", zmin=0.005)
    log("after ramp: cube", None if cc2 is None else np.round(cc2, 3).tolist(), "target", np.round(t, 3).tolist())

def obstacles_from(blobs, exclude=()):
    ex = set(id(b) for b in exclude)
    pts = []
    for b in blobs:
        if id(b) in ex:
            continue
        q = b["pts"]; q = q[np.isfinite(q[:, 2]) & (q[:, 2] > 0.05)]
        if len(q):
            pts.append(q[::max(1, len(q) // 200), :2])
    return np.concatenate(pts) if pts else np.zeros((0, 2))

def tool_push(robot, tool, vinfo, rgb, pw, cube, target, manner, obst=None):
    gcol = vinfo["grasp_color"]
    gxy, ztop, ax = grasp_pose_for(tool, gcol, rgb, pw)
    pts = tool["pts"]
    dists = np.linalg.norm(pts[:, :2] - gxy, axis=1)
    if manner == "ball":
        tip = np.array([gxy[0], gxy[1]]); tip_r = 0.02
    else:
        far = pts[dists > np.percentile(dists, 97)]
        tip = far[:, :2].mean(0); tip_r = 0.012
        # handle axis from grasp->tip
        u = tip - gxy; ax = u / (np.linalg.norm(u) + 1e-9)
    gz = max(ztop - 0.010, 0.012)
    # candidate grasp points along the handle colour region, scored by finger clearance
    cc = color_class(rgb)
    hm = tool["mask"] & (cc == gcol)
    hpts = pw[hm][:, :2] if hm.sum() >= 4 else np.array([gxy])
    others = []
    for b in ALL_BLOBS:
        if b is tool:
            continue
        q = b["pts"]; q = q[np.isfinite(q[:, 2]) & (q[:, 2] > 0.008)]
        if len(q):
            others.append(q[::max(1, len(q) // 300), :2])
    others = np.concatenate(others) if others else np.zeros((0, 2))
    perp = np.array([-ax[1], ax[0]])
    def clearance(g):
        if len(others) == 0:
            return 1.0
        f1 = g + perp * 0.032; f2 = g - perp * 0.032
        return float(min(np.min(np.linalg.norm(others - f1, axis=1)), np.min(np.linalg.norm(others - f2, axis=1)),
                         np.min(np.linalg.norm(others - g, axis=1)) + 0.02))
    sproj = (hpts - gxy) @ ax
    cands = [gxy + ax * s_ for s_ in np.linspace(sproj.min() + 0.01, sproj.max() - 0.01, 7)] if sproj.max() - sproj.min() > 0.03 else [gxy]
    cands.sort(key=lambda g: -min(clearance(g), 0.05) + 0.2 * np.linalg.norm(g - gxy))
    log("grasp cands", [(np.round(g, 3).tolist(), round(clearance(g), 3)) for g in cands[:3]])
    yaw = choose_grasp_yaw(ax, cur_yaw(robot.state))
    ok = False
    for gi, g in enumerate(cands[:3]):
        log("grasp try", gi, "xy", np.round(g, 3).tolist(), "gz", round(gz, 3), "tip", np.round(tip, 3).tolist())
        robot.gripper = PRE_GRIP
        robot.rot_hold = None
        robot.move_to([g[0], g[1], 0.15], speed=0.8, tol=0.01, max_steps=80)
        rotate_to_yaw(robot, yaw)
        robot.rot_hold = down_yaw(yaw)
        robot.move_to([g[0], g[1], gz + 0.05], speed=0.5, tol=0.004, max_steps=60)
        robot.move_to([g[0], g[1], gz], speed=0.25, tol=0.003, max_steps=50)
        robot.set_gripper(1.0, 15)
        w = robot.state[6] + robot.state[7]
        log("after close width", round(w, 4), "pos", np.round(robot.pos(), 3).tolist())
        if w > 0.004 and robot.pos()[2] < gz + 0.012:
            ok = True; break
        robot.set_gripper(-1.0, 8)
        robot.move_to([robot.pos()[0], robot.pos()[1], 0.15], speed=0.5, tol=0.01, max_steps=40)
        if robot.ended:
            return
    if not ok:
        log("grasp failed"); return
    grasp_tcp = robot.pos()
    robot.move_to([grasp_tcp[0], grasp_tcp[1], gz + 0.04], speed=0.2, tol=0.006, max_steps=40)
    robot.move_to([grasp_tcp[0], grasp_tcp[1], 0.18], speed=0.4, tol=0.008, max_steps=60)
    if robot.state[6] + robot.state[7] < 0.004:
        log("tool dropped after lift"); return
    if manner == "ball" or not FLAT:
        push_phase(robot, cube["center"], target["center"], manner, tip_r, obst)
    else:
        push_flat(robot, cube["center"], target["center"], tip - grasp_tcp[:2], gz, tip_r, obst)

def rot_about(axis, ang):
    axis = np.asarray(axis, float); axis = axis / (np.linalg.norm(axis) + 1e-12)
    return aa_to_mat(axis * ang)

def pitch_tool(robot, off, ang):
    """rotate gripper about horizontal axis perpendicular to the tool so the tip goes down."""
    a3 = np.array([off[0], off[1], 0.0]); a3 /= np.linalg.norm(a3) + 1e-12
    k = np.cross(a3, [0, 0, -1.0])
    Rt = rot_about(k, ang) @ aa_to_mat(robot.state[3:6])
    tgt = mat_to_aa(Rt)
    used = 0
    while used < 90 and not robot.ended:
        e = rot_error(robot.state[3:6], tgt)
        if np.linalg.norm(e) < 0.03:
            break
        st = np.clip(e / 0.5, -0.5, 0.5) * 0.7
        robot.step([[0, 0, 0, st[0], st[1], st[2], robot.gripper]] * 3)
        used += 3
    robot.rot_hold = tgt
    return tgt

PITCH = math.radians(40)
FLAT = True

def rot2(v, a):
    cs, sn = math.cos(a), math.sin(a)
    return np.array([cs * v[0] - sn * v[1], sn * v[0] + cs * v[1]])

def push_flat(robot, c, t, off0, gz, tip_r, obstacles=None):
    """tool kept near the table; tip position predicted from the grasp geometry (rigid yaw),
    refined by depth change near the expected tip."""
    c = np.asarray(c, float); t = np.asarray(t, float); off0 = np.asarray(off0, float)
    L0 = float(np.linalg.norm(off0))
    d = (t - c) / (np.linalg.norm(t - c) + 1e-9)
    phi = choose_push_rotation(off0, d, c, obstacles=obstacles)
    y0 = cur_yaw(robot.state)
    robot.move_to([robot.pos()[0], robot.pos()[1], 0.20], speed=0.3, tol=0.01, max_steps=40)
    # rotate at a comfortable spot between here and the pre-push pose
    p0g = c - d * 0.07
    tcp_pred = p0g - rot2(off0, phi)
    mid = 0.5 * (robot.pos()[:2] + tcp_pred)
    v = mid - np.asarray(BASE); rr = np.linalg.norm(v)
    mid = np.asarray(BASE) + v / (rr + 1e-9) * float(np.clip(rr, 0.35, 0.60))
    robot.move_to([mid[0], mid[1], 0.20], speed=0.4, tol=0.01, max_steps=100)
    robot.rot_hold = None
    rotate_to_yaw(robot, wrap(y0 + phi), max_steps=240, gain=0.3)
    dphi = wrap(cur_yaw(robot.state) - y0)
    off = rot2(off0, dphi)
    robot.rot_hold = down_yaw(cur_yaw(robot.state))
    log("flat: rotated", round(dphi, 3), "wanted", round(phi, 3), "off", np.round(off, 3).tolist(), "L", round(L0, 3))
    if robot.state[6] + robot.state[7] < 0.004:
        log("tool dropped"); return
    push_z = gz + 0.012
    for attempt in range(4):
        if robot.ended:
            break
        dv = (t - c); Ld = np.linalg.norm(dv)
        if Ld < 0.012:
            break
        d = dv / Ld
        p0 = c - d * (0.02 + tip_r + 0.03)
        p1 = t - d * (0.02 + tip_r - 0.004)
        tcp0 = p0 - off; tcp1 = p1 - off
        log("flat push", attempt, "cube", np.round(c, 3).tolist(), "p0", np.round(p0, 3).tolist(), "p1", np.round(p1, 3).tolist(), "tcp0", np.round(tcp0, 3).tolist())
        robot.move_to([tcp0[0], tcp0[1], 0.20], speed=0.5, tol=0.008, max_steps=120)
        ok = robot.move_to([tcp0[0], tcp0[1], push_z + 0.03], speed=0.3, tol=0.006, max_steps=80)
        ok = robot.move_to([tcp0[0], tcp0[1], push_z], speed=0.15, tol=0.005, max_steps=40) and ok
        log("at pre-push", ok, np.round(robot.pos(), 3).tolist(), "step", robot.step_id)
        if not ok and np.linalg.norm(robot.pos()[:2] - tcp0) > 0.03:
            log("pre-push unreachable")
            robot.move_to([robot.pos()[0], robot.pos()[1], 0.20], speed=0.4, tol=0.01, max_steps=40)
            break
        for it in range(2):
            tip, tzm = find_tip_near(robot, robot.pos()[:2] + off, radius=0.06)
            if tip is None:
                break
            noff = tip - robot.pos()[:2]
            if abs(np.linalg.norm(noff) - L0) > 0.06 or np.linalg.norm(noff - off) > 0.05:
                log("tip estimate rejected", np.round(noff, 3).tolist()); break
            off = noff; tcp0 = p0 - off; tcp1 = p1 - off
            if np.linalg.norm(tip - p0) < 0.008:
                break
            robot.move_to([tcp0[0], tcp0[1], push_z], speed=0.12, tol=0.004, max_steps=40)
        robot.move_to([tcp1[0], tcp1[1], push_z], speed=0.10, tol=0.003, max_steps=300, stall_steps=40)
        if robot.ended:
            break
        robot.move_to([robot.pos()[0] - d[0] * 0.02, robot.pos()[1] - d[1] * 0.02, push_z + 0.005], speed=0.2, tol=0.006, max_steps=20)
        robot.move_to([robot.pos()[0], robot.pos()[1], 0.20], speed=0.4, tol=0.01, max_steps=60)
        p, rgb2, pw2 = scene_now(robot)
        cc2 = color_center(rgb2, pw2, "red", zmin=0.005)
        if cc2 is None:
            log("cube lost"); break
        c = cc2
        log("cube now", np.round(c, 3).tolist(), "err", round(float(np.linalg.norm(c - t)), 4), "step", robot.step_id)
        if np.linalg.norm(c - t) < 0.012:
            break
        if robot.state[6] + robot.state[7] < 0.004:
            log("tool dropped"); break
    log("final cube", np.round(c, 3).tolist(), "target", np.round(t, 3).tolist())

def push_phase(robot, c, t, manner, tip_r, obstacles=None):
    c = np.asarray(c, float); t = np.asarray(t, float)
    if manner == "ball":
        off = np.zeros(2); dz = 0.0
        robot.rot_hold = down_yaw(cur_yaw(robot.state))
    else:
        tip, tz = measure_tip(robot)
        if tip is None:
            log("no tip; abort"); return
        off = tip - robot.pos()[:2]; dz = robot.pos()[2] - tz
        d = (t - c) / (np.linalg.norm(t - c) + 1e-9)
        phi = choose_push_rotation(off * 0.6, d, c, obstacles=obstacles)
        y0 = cur_yaw(robot.state)
        robot.rot_hold = None
        rotate_to_yaw(robot, wrap(y0 + phi), max_steps=120)
        dphi = wrap(cur_yaw(robot.state) - y0)
        cs, sn = math.cos(dphi), math.sin(dphi)
        off = np.array([cs * off[0] - sn * off[1], sn * off[0] + cs * off[1]])
        robot.rot_hold = down_yaw(cur_yaw(robot.state))
        robot.move_to([robot.pos()[0], robot.pos()[1], 0.32], speed=0.5, tol=0.01, max_steps=40)
        pitch_tool(robot, off, PITCH)
        tip, tz = measure_tip(robot)
        if tip is None:
            log("no tip after rotate; abort"); return
        off = tip - robot.pos()[:2]; dz = robot.pos()[2] - tz
        log("rotated by", round(dphi, 3), "wanted", round(phi, 3), "off", np.round(off, 3).tolist(), "dz", round(dz, 3))
    push_z = max(0.016 + dz, 0.02)
    for attempt in range(4):
        if robot.ended:
            break
        d = (t - c); L = np.linalg.norm(d)
        if L < 0.012:
            break
        d = d / L
        p0 = c - d * (0.02 + tip_r + 0.035)
        p1 = t - d * (0.02 + tip_r - 0.006)
        tcp0 = p0 - off; tcp1 = p1 - off
        log("push", attempt, "cube", np.round(c, 3).tolist(), "tip p0", np.round(p0, 3).tolist(), "p1", np.round(p1, 3).tolist(), "tcp0", np.round(tcp0, 3).tolist())
        ok = robot.move_to([tcp0[0], tcp0[1], push_z + 0.07], speed=0.5, tol=0.006)
        ok = robot.move_to([tcp0[0], tcp0[1], push_z], speed=0.2, tol=0.005, max_steps=60) and ok
        log("at pre-push", ok, np.round(robot.pos(), 3).tolist(), "step", robot.step_id)
        if not ok and np.linalg.norm(robot.pos()[:2] - tcp0) > 0.03:
            log("pre-push unreachable; lifting")
            robot.move_to([robot.pos()[0], robot.pos()[1], push_z + 0.1], speed=0.4, tol=0.01, max_steps=30)
            break
        if manner != "ball":
            for it in range(3):
                tip, tzm = find_tip_near(robot, p0, radius=0.09)
                if tip is None:
                    break
                off = tip - robot.pos()[:2]
                if tzm is not None and tzm > 0.03:
                    push_z -= min(0.03, tzm - 0.012)
                tcp0 = p0 - off; tcp1 = p1 - off
                if np.linalg.norm(tip - p0) < 0.008 and (tzm is None or tzm < 0.03):
                    break
                robot.move_to([tcp0[0], tcp0[1], push_z], speed=0.15, tol=0.004, max_steps=40)
        robot.move_to([tcp1[0], tcp1[1], push_z], speed=0.10, tol=0.003, max_steps=300, stall_steps=40)
        if robot.ended:
            break
        robot.move_to([robot.pos()[0] - d[0] * 0.025, robot.pos()[1] - d[1] * 0.025, robot.pos()[2] + 0.01], speed=0.2, tol=0.006, max_steps=20)
        robot.move_to([robot.pos()[0], robot.pos()[1], push_z + 0.12], speed=0.4, tol=0.01, max_steps=60)
        p, rgb2, pw2 = scene_now(robot)
        cc2 = color_center(rgb2, pw2, "red", zmin=0.005)
        if cc2 is None:
            log("cube lost"); break
        c = cc2
        log("cube now", np.round(c, 3).tolist(), "err", round(float(np.linalg.norm(c - t)), 4), "step", robot.step_id)
        if np.linalg.norm(c - t) < 0.012:
            break
        if manner != "ball":
            tip, tz = measure_tip(robot)
            if tip is not None and abs(np.linalg.norm(tip - robot.pos()[:2]) - np.linalg.norm(off)) < 0.04:
                off = tip - robot.pos()[:2]; dz = robot.pos()[2] - tz
    log("final cube", np.round(c, 3).tolist(), "target", np.round(t, 3).tolist())


def main(ctx):
    try:
        _main(ctx)
    except Exception:
        log("controller exception:\n" + traceback.format_exc())
