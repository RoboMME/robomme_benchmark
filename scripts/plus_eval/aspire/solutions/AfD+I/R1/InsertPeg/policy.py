# InsertPeg v01_stall_search -- ASPIRE task worker, robomme_plus/robomme_plus/task_3.
#
# v01 change (trace-guided repair of the v00 episode-0 failure, see findings.md):
#   v00 re-enacted the video robot states correctly up to contact, but the peg tip stalled
#   on the box face ~3.3 cm short of the video's final depth; the open-loop replay kept
#   advancing joint targets (wind-up), which rotated the peg in hand. v01 (1) detects the
#   insertion phase from the video itself, never skips frames there, (2) monitors live
#   progress along the video's insertion direction and stops the replay on a stall
#   (no wind-up), then (3) runs a bounded closed-loop back-off + lateral/vertical search
#   with slow pushes until the live progress passes the stall point, and finally pushes to
#   the video's own final depth. Everything before contact is unchanged from v00.
#
# Strategy (hypothesis from the released recordings; see task_analysis.md):
#   The episode's conditioning video shows a robot grasping one peg at one end and
#   inserting it into one side of the box; the execution scene then starts from the same
#   reset state. The conditioning video is per-episode input and retains, at each frame,
#   the recorded robot state (joints, gripper, end-effector pose). The primary policy
#   re-enacts that recorded robot-state sequence tick by tick in the live episode
#   (static frames skipped, gripper binarised, arm settled before each gripper
#   transition), after checking that the live robot and scene match the video start.
#   Nothing is taken from released recordings or hard-coded per episode.
#   Fallback (video has no usable robot states): perception-only grasp of the peg that
#   the video depth shows was removed, lift, and push into the box side seen in the video.
#
# Heavily instrumented: this single run is also the scene probe.
import math

result = {"version": "v01_stall_search", "errors": [], "phases": {}}
R = result
EP = {"ended": False, "steps": 0, "term": False, "trunc": False, "end_reason": None,
      "consec_exc": 0, "last_q": None, "last_g": None, "flags_seen": None,
      "last_p": None}


def note_error(phase, exc):
    R["errors"].append({"phase": phase, "error": str(exc)[:400]})


def flist(values):
    return [float(v) for v in values]


def r4(v):
    return round(float(v), 4)


def r4list(values):
    return [round(float(v), 4) for v in values]


def as_grid(arr):
    """2-D array -> list of lists of floats (fast path tolist)."""
    try:
        return arr.tolist()
    except Exception:
        return [[float(v) for v in row] for row in arr]


def quat_to_mat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def terminal_flags(reply):
    term = False
    trunc = False
    seen = []
    try:
        term = bool(reply.terminated)
        seen.append("terminated")
    except Exception:
        pass
    try:
        trunc = bool(reply.truncated)
        seen.append("truncated")
    except Exception:
        pass
    if not seen:
        try:
            term = bool(reply.done)
            seen.append("done")
        except Exception:
            pass
    return term, trunc, seen


def live_from_reply(reply):
    """Joint/gripper reading from a step reply's observation (no extra call)."""
    try:
        rs = reply.observation.robot_state
        return flist(rs.joint_positions["primary"])[:7], float(rs.gripper_positions["primary"])
    except Exception:
        return None, None


EMB = {"name": "robomme_plus"}


def send(q, g, phase):
    """One control tick: absolute joint target + normalized gripper target."""
    if EP["ended"]:
        return None
    try:
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=[float(v) for v in q],
            gripper_position=float(g), embodiment=EMB["name"])}))
    except Exception as exc:
        msg = str(exc)
        note_error("step:" + phase, exc)
        EP["consec_exc"] += 1
        if "ended" in msg or "closing" in msg or EP["consec_exc"] >= 3:
            EP["ended"] = True
            EP["end_reason"] = "exception:" + msg[:120]
        return None
    EP["consec_exc"] = 0
    EP["steps"] += 1
    term, trunc, seen = terminal_flags(reply)
    if EP["flags_seen"] is None:
        EP["flags_seen"] = seen
        try:
            R["first_step_reply"] = {"flags": seen, "error": str(reply.error)[:200]}
        except Exception:
            R["first_step_reply"] = {"flags": seen}
    lq, lg = live_from_reply(reply)
    if lq is not None:
        EP["last_q"] = lq
        EP["last_g"] = lg
    try:
        lp, lqq = ee_of(reply.observation.robot_state)
        EP["last_p"] = lp
    except Exception:
        EP["last_p"] = None
    if term or trunc:
        EP["ended"] = True
        EP["term"] = term
        EP["trunc"] = trunc
        EP["end_reason"] = "terminated" if term else "truncated"
        R["terminal_at"] = {"phase": phase, "tick": EP["steps"], "terminated": term,
                            "truncated": trunc}
    return reply


def joint_err(q):
    if EP["last_q"] is None:
        return None
    return max(abs(a - b) for a, b in zip(EP["last_q"], q))


def hold(q, g, max_ticks, tol, phase):
    """Hold a joint target until the measured joints converge (bounded ticks)."""
    used = 0
    while used < max_ticks and not EP["ended"]:
        err = joint_err(q)
        if err is not None and err < tol and used > 0:
            break
        send(q, g, phase)
        used += 1
    return used


def ee_of(rs):
    try:
        pose = rs.end_effector_poses["primary"]
        return flist(pose.position), flist(pose.quaternion_wxyz)
    except Exception:
        return None, None


def dist3(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------- Phase 0: metadata
try:
    ctx = get_task_context()
    R["goal"] = str(ctx.language)
    R["task_name"] = str(ctx.task_name)
    try:
        R["family"] = str(ctx.family)
    except Exception:
        pass
except Exception as exc:
    note_error("context", exc)
MAX_STEPS = 1000
HZ = 20.0
try:
    meta = robomme_plus.get_task_metadata()
    R["task_metadata"] = {str(k): str(meta[k])[:300] for k in meta}
    try:
        MAX_STEPS = int(meta["max_steps"])
    except Exception:
        pass
    try:
        HZ = float(meta["control_hz"])
    except Exception:
        try:
            HZ = float(meta["control_frequency"])
        except Exception:
            pass
except Exception as exc:
    note_error("task_metadata", exc)
try:
    cmeta = robomme_plus.get_controller_metadata()
    R["controller_metadata"] = {str(k): str(cmeta[k])[:300] for k in cmeta}
except Exception as exc:
    note_error("controller_metadata", exc)
R["max_steps_used"] = MAX_STEPS
R["hz_used"] = HZ

BASE = "robot_base"
live_obs = None
live_state = None
try:
    live_state = get_robot_state()
    BASE = str(live_state.base_frame)
    EMB["name"] = str(live_state.embodiment)
    EP["last_q"] = flist(live_state.joint_positions["primary"])[:7]
    EP["last_g"] = float(live_state.gripper_positions["primary"])
    p0, q0 = ee_of(live_state)
    R["live_start"] = {"base_frame": BASE, "embodiment": EMB["name"],
                       "joints": r4list(EP["last_q"]), "gripper": r4(EP["last_g"]),
                       "ee_pos": r4list(p0) if p0 else None,
                       "ee_quat": r4list(q0) if q0 else None,
                       "timestamp_s": r4(live_state.timestamp_s)}
except Exception as exc:
    note_error("robot_state", exc)
CAMS = []
try:
    live_obs = get_observation()
    for name in live_obs.cameras:
        CAMS.append(str(name))
    R["cameras"] = {}
    for name in CAMS:
        cam = live_obs.cameras[name]
        info = {}
        try:
            info["rgb_shape"] = [int(v) for v in cam.rgb.shape]
            info["intrinsics"] = [r4list(row) for row in cam.intrinsics]
            info["pose_pos"] = r4list(cam.camera_pose.position)
            info["pose_quat"] = r4list(cam.camera_pose.quaternion_wxyz)
            info["pose_frame"] = str(cam.camera_pose.frame)
        except Exception as exc:
            note_error("camera_info:" + name, exc)
        R["cameras"][name] = info
except Exception as exc:
    note_error("observation", exc)
FRONT = "front" if "front" in CAMS else (CAMS[0] if CAMS else "front")

# ---------------------------------------------------------------- Phase 1: conditioning video
vq = []
vg = []
vt = []
vee = []
vquat = []
vframes = []
try:
    video = robomme_plus.get_conditioning_video()
    vframes = list(video)
    for frame in vframes:
        try:
            rs = frame.robot_state
            jraw = flist(rs.joint_positions["primary"])
            if not vq:
                R["video_joint_len"] = len(jraw)
            vq.append(jraw[:7] if len(jraw) >= 7 else jraw)
            vg.append(float(rs.gripper_positions["primary"]))
            pos, quat = ee_of(rs)
            vee.append(pos)
            vquat.append(quat)
        except Exception:
            vq.append(None)
            vg.append(None)
            vee.append(None)
            vquat.append(None)
        try:
            vt.append(float(frame.timestamp_s))
        except Exception:
            vt.append(None)
except Exception as exc:
    note_error("conditioning_video", exc)
NV = len(vframes)
valid_idx = [i for i in range(NV) if vq[i] is not None and vg[i] is not None and len(vq[i]) == 7]
vid = {"frames": NV, "frames_with_state": len(valid_idx)}
if NV > 0:
    vid["t_first"] = vt[0]
    vid["t_last"] = vt[NV - 1]
    dts = [vt[i + 1] - vt[i] for i in range(NV - 1) if vt[i] is not None and vt[i + 1] is not None]
    if dts:
        vid["dt_min"] = r4(min(dts))
        vid["dt_max"] = r4(max(dts))
if valid_idx:
    gl = [vg[i] for i in valid_idx]
    vid["grip_min"] = r4(min(gl))
    vid["grip_max"] = r4(max(gl))
    vid["grip_profile_every5"] = [r4(vg[i]) for i in valid_idx[::5]]
    vid["ee_every10"] = [r4list(vee[i]) for i in valid_idx[::10] if vee[i] is not None]
    vid["joints_first"] = r4list(vq[valid_idx[0]])
    vid["joints_last"] = r4list(vq[valid_idx[-1]])
try:
    vid["sampling_meta"] = R.get("task_metadata", {}).get("conditioning_video_sampling")
except Exception:
    pass
R["video"] = vid

# ---------------------------------------------------------------- Phase 2: scene diagnostics (no ticks)
diag = {}
R["scene_diag"] = diag
live_depth = None
v0_depth = None
vlast_depth = None
try:
    live_depth = as_grid(live_obs.cameras[FRONT].depth_m)
except Exception as exc:
    note_error("live_depth", exc)
if NV > 0:
    try:
        v0_depth = as_grid(vframes[0].cameras[FRONT].depth_m)
        vlast_depth = as_grid(vframes[NV - 1].cameras[FRONT].depth_m)
        cp0 = vframes[0].cameras[FRONT].camera_pose
        diag["video_cam_pos"] = r4list(cp0.position)
        diag["video_cam_quat"] = r4list(cp0.quaternion_wxyz)
        try:
            lp = live_obs.cameras[FRONT].camera_pose
            diag["cam_pose_delta_m"] = r4(dist3(cp0.position, lp.position))
            diag["cam_quat_dot"] = r4(abs(sum(float(a) * float(b) for a, b in
                                              zip(cp0.quaternion_wxyz, lp.quaternion_wxyz))))
        except Exception as exc:
            note_error("cam_pose_compare", exc)
    except Exception as exc:
        note_error("video_depth", exc)

H = 0
W = 0
if live_depth is not None:
    H = len(live_depth)
    W = len(live_depth[0]) if H else 0
# live vs video-start depth: does the execution scene equal the video's reset scene?
if live_depth is not None and v0_depth is not None and len(v0_depth) == H:
    changed = 0
    big = 0
    total = 0
    for y in range(H):
        rowl = live_depth[y]
        rowv = v0_depth[y]
        for x in range(W):
            d = abs(rowl[x] - rowv[x])
            total += 1
            if d > 0.006:
                changed += 1
            if d > 0.05:
                big += 1
    diag["live_vs_video0_changed_px"] = changed
    diag["live_vs_video0_big_px"] = big
    diag["px_total"] = total
# removed-object blob: depth increased by a peg-like amount between video start and end
blob = []
newpx = []
if v0_depth is not None and vlast_depth is not None and len(vlast_depth) == len(v0_depth):
    for y in range(len(v0_depth)):
        r0 = v0_depth[y]
        r1 = vlast_depth[y]
        for x in range(len(r0)):
            inc = r1[x] - r0[x]
            if 0.004 < inc < 0.05:
                blob.append((x, y))
            elif inc < -0.02:
                newpx.append((x, y))
    if blob:
        diag["removed_blob_px"] = len(blob)
        diag["removed_blob_center"] = [r4(sum(p[0] for p in blob) / len(blob)),
                                       r4(sum(p[1] for p in blob) / len(blob))]
        xs = [p[0] for p in blob]
        ys = [p[1] for p in blob]
        diag["removed_blob_bbox"] = [min(xs), min(ys), max(xs), max(ys)]
    diag["video_end_new_px"] = len(newpx)

# segmentation probes on the live front image (instrumentation + fallback)
seg_store = {}


def probe(prompt):
    out = {"prompt": prompt, "ok": False, "n": 0, "cands": []}
    try:
        found = segment_text(FRONT, prompt)
        out["ok"] = bool(found.ok)
        if found.ok:
            out["n"] = len(found.segmentations)
            seg_store[prompt] = found
            for cand in found.segmentations[:6]:
                c = {"score": r4(cand.score)}
                try:
                    if cand.box_xyxy is not None:
                        c["box"] = r4list(cand.box_xyxy)
                except Exception:
                    pass
                try:
                    c["label"] = str(cand.label)
                except Exception:
                    pass
                out["cands"].append(c)
    except Exception as exc:
        out["error"] = str(exc)[:200]
    return out


R["probes"] = []
for prompt in ("peg", "box", "small rectangular block", "cube"):
    R["probes"].append(probe(prompt))


def points_list(cloud):
    try:
        return cloud.points.tolist()
    except Exception:
        return [flist(p) for p in cloud.points]


def cand_mask(cand):
    try:
        return as_grid(cand.mask)
    except Exception:
        return None


def box_center_of(cand):
    try:
        b = [float(v) for v in cand.box_xyxy]
        return [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, b]
    except Exception:
        return None


# which live peg candidate overlaps the removed blob (pixel space; camera static)
PEG_CHOICE = {"index": None}
try:
    if "peg" in seg_store and blob:
        blob_set = set(blob)
        best = None
        best_ov = 0
        overlaps = []
        for ci, cand in enumerate(seg_store["peg"].segmentations[:10]):
            m = cand_mask(cand)
            ov = 0
            if m is not None:
                for (x, y) in blob:
                    if m[y][x]:
                        ov += 1
            overlaps.append(ov)
            if ov > best_ov:
                best_ov = ov
                best = ci
        PEG_CHOICE["index"] = best
        PEG_CHOICE["overlaps"] = overlaps
    R["peg_choice"] = PEG_CHOICE
except Exception as exc:
    note_error("peg_choice", exc)

# side of box seen at video end: new near-box pixels left/right of the live box centre
try:
    if "box" in seg_store and newpx:
        bc = box_center_of(seg_store["box"].segmentations[0])
        if bc is not None:
            bw = max(4.0, bc[2][2] - bc[2][0])
            bh = max(4.0, bc[2][3] - bc[2][1])
            near = [p for p in newpx if abs(p[1] - bc[1]) < bh and abs(p[0] - bc[0]) < 3.0 * bw]
            if near:
                cx = sum(p[0] for p in near) / len(near)
                diag["video_end_near_box_px"] = len(near)
                diag["video_end_near_box_cx"] = r4(cx)
                diag["box_center_px"] = [r4(bc[0]), r4(bc[1])]
                diag["side_image"] = "right" if cx > bc[0] else "left"
except Exception as exc:
    note_error("side_diag", exc)

# camera convention calibration: manual unprojection vs mask_to_point_cloud on the box mask
CONV = {"name": None}


def cam_point(conv, u, v, d, K):
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    a = (u - cx) / fx * d
    b = (v - cy) / fy * d
    if conv == "opencv":
        return [a, b, d]
    if conv == "opengl":
        return [a, -b, -d]
    return [d, -a, -b]  # x-forward, y-left, z-up


def to_base(conv, u, v, d, K, Rm, t):
    pc = cam_point(conv, u, v, d, K)
    return [Rm[i][0] * pc[0] + Rm[i][1] * pc[1] + Rm[i][2] * pc[2] + t[i] for i in range(3)]


try:
    if "box" in seg_store and live_depth is not None:
        cand = seg_store["box"].segmentations[0]
        m = cand_mask(cand)
        CONV["mask_attr_ok"] = m is not None
        cloud = mask_to_point_cloud(cand, FRONT, target_frame=BASE)
        pts = points_list(cloud)
        if m is not None and len(pts) > 5:
            api_c = [sum(p[i] for p in pts) / len(pts) for i in range(3)]
            camf = live_obs.cameras[FRONT]
            K = [flist(row) for row in camf.intrinsics]
            Rm = quat_to_mat(camf.camera_pose.quaternion_wxyz)
            t = flist(camf.camera_pose.position)
            pix = []
            for y in range(0, H, 1):
                row = m[y]
                for x in range(W):
                    if row[x] and live_depth[y][x] > 0:
                        pix.append((x, y, live_depth[y][x]))
            errs = {}
            for conv in ("opencv", "opengl", "xfwd"):
                acc = [0.0, 0.0, 0.0]
                for (x, y, d) in pix:
                    p = to_base(conv, x + 0.5, y + 0.5, d, K, Rm, t)
                    acc[0] += p[0]
                    acc[1] += p[1]
                    acc[2] += p[2]
                if pix:
                    c = [acc[i] / len(pix) for i in range(3)]
                    errs[conv] = r4(dist3(c, api_c))
            best = None
            for k in errs:
                if best is None or errs[k] < errs[best]:
                    best = k
            CONV["name"] = best
            CONV["errors_m"] = errs
            CONV["api_box_centroid"] = r4list(api_c)
            CONV["box_cloud_points"] = len(pts)
            try:
                geo = estimate_geometry(cloud)
                CONV["box_geom_center"] = r4list(geo.pose.position)
                CONV["box_geom_extents"] = r4list(geo.extents)
            except Exception as exc:
                note_error("box_geometry", exc)
    R["camera_convention"] = CONV
except Exception as exc:
    note_error("convention", exc)

# 3-D location of the removed blob in the video frame and of the video-end insertion stuff
try:
    if CONV["name"] is not None and blob and NV > 0:
        camv = vframes[0].cameras[FRONT]
        K = [flist(row) for row in camv.intrinsics]
        Rm = quat_to_mat(camv.camera_pose.quaternion_wxyz)
        t = flist(camv.camera_pose.position)
        bp = [to_base(CONV["name"], x + 0.5, y + 0.5, v0_depth[y][x], K, Rm, t) for (x, y) in blob]
        diag["removed_blob_base"] = r4list([sum(p[i] for p in bp) / len(bp) for i in range(3)])
except Exception as exc:
    note_error("blob3d", exc)

# ---------------------------------------------------------------- Phase 3: build replay plan
use_replay = len(valid_idx) >= 20
INS = {"ok": False}
R["policy_branch"] = "video_robot_state_replay" if use_replay else "perception_fallback"
plan = []
plan_info = {}
if use_replay:
    try:
        idx = valid_idx
        g_open = max(vg[i] for i in idx)
        g_low = min(vg[i] for i in idx)
        thr = g_open - 0.1 * (g_open - g_low)
        if g_open - g_low < 0.001:
            thr = g_low - 1.0  # gripper never moved in the video: keep it open
        plan_info["g_threshold"] = r4(thr)
        # v01: insertion phase from the video itself: the final contiguous run of held
        # frames at the final height (|dz|<6 mm) within 8 cm (xy) of the final EE.
        try:
            i_end = valid_idx[-1]
            p_end = vee[i_end]
            if p_end is not None:
                i_ins = None
                for i in reversed(valid_idx):
                    pe = vee[i]
                    if pe is None or vg[i] >= thr:
                        break
                    dxy = math.sqrt((pe[0] - p_end[0]) ** 2 + (pe[1] - p_end[1]) ** 2)
                    if abs(pe[2] - p_end[2]) < 0.006 and dxy < 0.08:
                        i_ins = i
                    else:
                        break
                if i_ins is not None:
                    p_ins = vee[i_ins]
                    dvx = p_end[0] - p_ins[0]
                    dvy = p_end[1] - p_ins[1]
                    nrm = math.sqrt(dvx * dvx + dvy * dvy)
                    if nrm > 0.01:
                        INS["ok"] = True
                        INS["i_ins"] = i_ins
                        INS["i_end"] = i_end
                        INS["p_ins"] = p_ins
                        INS["p_end"] = p_end
                        INS["u"] = [dvx / nrm, dvy / nrm, 0.0]
                        INS["s"] = [-dvy / nrm, dvx / nrm, 0.0]
                        INS["prog_end"] = nrm
                        INS["quat_end"] = vquat[i_end]
            plan_info["insertion"] = {"ok": INS["ok"], "i_ins": INS.get("i_ins"),
                                      "u": r4list(INS["u"]) if INS["ok"] else None,
                                      "prog_end_m": r4(INS["prog_end"]) if INS["ok"] else None,
                                      "p_ins": r4list(INS["p_ins"]) if INS["ok"] else None}
        except Exception as exc:
            note_error("insertion_detect", exc)
        LOOK = 2
        cmd = {}
        for k in range(len(idx)):
            j = idx[min(k + LOOK, len(idx) - 1)]
            cmd[idx[k]] = 0.0 if vg[j] < thr else 1.0
        transitions = []
        last_q = vq[idx[0]]
        last_g = vg[idx[0]]
        last_c = cmd[idx[0]]
        for k in range(1, len(idx)):
            i = idx[k]
            dq = max(abs(a - b) for a, b in zip(vq[i], last_q))
            dg = abs(vg[i] - last_g)
            in_ins = INS["ok"] and i >= INS["i_ins"]
            if dq < (0.0002 if in_ins else 0.001) and dg < 0.01 and cmd[i] == last_c:
                continue
            ticks = 1
            if vt[i] is not None and vt[idx[k - 1]] is not None:
                ticks = max(1, int(round((vt[i] - vt[idx[k - 1]]) * HZ)))
            ticks = max(ticks, int(math.ceil(dq / 0.06)))
            ticks = min(ticks, 40)
            for s in range(1, ticks + 1):
                f = s / float(ticks)
                qi = [a + (b - a) * f for a, b in zip(last_q, vq[i])]
                plan.append((qi, cmd[i], i))
            if cmd[i] != last_c:
                transitions.append({"src": i, "to": cmd[i], "plan_index": len(plan) - ticks})
            last_q = vq[i]
            last_g = vg[i]
            last_c = cmd[i]
        plan_info["g_open"] = r4(g_open)
        plan_info["transitions"] = transitions
        plan_info["planned_ticks_raw"] = len(plan)
        budget = MAX_STEPS - 250  # v01: reserve ticks for the insertion search
        if len(plan) > budget and budget > 50:
            keep_every = int(math.ceil(len(plan) / float(budget)))
            thinned = []
            for k in range(len(plan)):
                is_edge = k == len(plan) - 1 or (k > 0 and plan[k][1] != plan[k - 1][1])
                if INS["ok"] and plan[k][2] >= INS["i_ins"]:
                    is_edge = True  # never thin the insertion phase
                if k % keep_every == 0 or is_edge:
                    thinned.append(plan[k])
            plan = thinned
            plan_info["thinned_every"] = keep_every
        plan_info["planned_ticks"] = len(plan)
    except Exception as exc:
        note_error("plan", exc)
        use_replay = False
        R["policy_branch"] = "perception_fallback(plan_error)"
R["plan"] = plan_info

# ---------------------------------------------------------------- Phase 4: execute replay
log = []
events = []
R["events"] = events
STALL = {"hit": False}


def live_p():
    """Live EE position: from the last step reply, else one state query."""
    if EP["last_p"] is not None:
        return EP["last_p"]
    try:
        p, qq = ee_of(get_robot_state())
        return p
    except Exception as exc:
        note_error("live_p", exc)
        return None


def prog_of(p):
    """Progress (m) along the video's horizontal insertion direction from p_ins."""
    a = INS["p_ins"]
    u = INS["u"]
    return (p[0] - a[0]) * u[0] + (p[1] - a[1]) * u[1]


def wrist_depth_grid(tag):
    """Coarse (every 16 px) wrist depth for the record at a contact event."""
    try:
        ob = get_observation()
        d = as_grid(ob.cameras["wrist"].depth_m)
        R["wrist_depth_" + tag] = [[r4(d[y][x]) for x in range(0, len(d[0]), 16)]
                                   for y in range(0, len(d), 16)]
    except Exception as exc:
        note_error("wrist_depth:" + tag, exc)


if use_replay and plan:
    try:
        q_first = vq[valid_idx[0]]
        g_first = 1.0
        err0 = joint_err(q_first)
        events.append({"ev": "start_align", "joint_err": r4(err0) if err0 is not None else None})
        if err0 is not None and err0 > 0.01:
            n_al = min(60, max(2, int(math.ceil(err0 / 0.04))))
            start_q = list(EP["last_q"])
            for s in range(1, n_al + 1):
                f = s / float(n_al)
                send([a + (b - a) * f for a, b in zip(start_q, q_first)], g_first, "align")
            hold(q_first, g_first, 15, 0.005, "align_hold")
        prev_c = plan[0][1] if plan else 1.0
        prev_q = q_first
        lag_hist = []
        prog_hist = []
        ins_log = []
        for k in range(len(plan)):
            if EP["ended"]:
                break
            q, c, src = plan[k]
            if c != prev_c:
                used = hold(prev_q, prev_c, 12, 0.004, "pre_gripper_settle")
                ev = {"ev": "gripper_" + ("close" if c < 0.5 else "open"), "src": src,
                      "tick": EP["steps"], "settle_ticks": used,
                      "joint_err": r4(joint_err(prev_q) or 0.0)}
                try:
                    if not EP["ended"]:
                        rs = get_robot_state()
                        p, qq = ee_of(rs)
                        ev["live_ee"] = r4list(p)
                        if vee[src] is not None:
                            ev["video_ee"] = r4list(vee[src])
                            ev["ee_delta_m"] = r4(dist3(p, vee[src]))
                except Exception as exc:
                    note_error("event_state", exc)
                events.append(ev)
            send(q, c, "replay")
            if c < 0.5 and prev_c >= 0.5:
                # let the jaw finish closing on the peg before the arm moves on
                hist = []
                for wt in range(15):
                    if EP["ended"]:
                        break
                    send(q, 0.0, "close_wait")
                    hist.append(EP["last_g"])
                    if len(hist) >= 3 and hist[-1] is not None and hist[-3] is not None \
                            and abs(hist[-1] - hist[-3]) < 0.004:
                        break
                events.append({"ev": "close_done", "tick": EP["steps"],
                               "live_grip": r4(EP["last_g"] or 0.0),
                               "video_grip_min": r4(min(vg[i] for i in valid_idx)),
                               "wait_ticks": len(hist)})
            if k % 10 == 0:
                e = joint_err(q)
                log.append([EP["steps"], src, r4(e) if e is not None else None,
                            r4(EP["last_g"] or 0.0)])
            # v01: stall monitor in the insertion phase (stops target wind-up on contact)
            if INS["ok"] and src >= INS["i_ins"] and c < 0.5 and vee[src] is not None:
                lp = live_p()
                if lp is not None:
                    vprog = prog_of(vee[src])
                    lprog = prog_of(lp)
                    lag_hist.append(vprog - lprog)
                    prog_hist.append(lprog)
                    ins_log.append([EP["steps"], src, r4(vprog), r4(lprog)])
                    if len(prog_hist) >= 4:
                        moved = prog_hist[-1] - prog_hist[-4]
                        if min(lag_hist[-3:]) > 0.005 and moved < 0.001:
                            STALL["hit"] = True
                            STALL["tick"] = EP["steps"]
                            STALL["src"] = src
                            STALL["p"] = lp
                            STALL["prog"] = lprog
                            STALL["video_prog"] = vprog
                            STALL["q"] = list(EP["last_q"])
                            events.append({"ev": "insertion_stall", "tick": EP["steps"],
                                           "src": src, "live_ee": r4list(lp),
                                           "video_ee": r4list(vee[src]),
                                           "live_prog": r4(lprog), "video_prog": r4(vprog),
                                           "prog_end": r4(INS["prog_end"]),
                                           "grip": r4(EP["last_g"] or 0.0)})
                            break
            prev_c = c
            prev_q = q
        R["replay_log_every10"] = log
        R["insertion_log"] = ins_log
        if not EP["ended"]:
            if STALL["hit"]:
                # hold the measured joints (not the wound-up target) to relax contact
                hold(STALL["q"], 0.0, 3, 0.0005, "stall_relax")
                wrist_depth_grid("at_stall")
            else:
                used = hold(prev_q, prev_c, 20, 0.003, "final_hold")
                events.append({"ev": "final_hold", "ticks": used, "tick": EP["steps"]})
    except Exception as exc:
        note_error("replay", exc)

    # v01: closed-loop insertion -- back off, shift laterally / vertically, push slowly
    SEARCH = {"trials": [], "inserted": False}
    R["search"] = SEARCH
    try:
        need = False
        lp0 = None
        if INS["ok"] and not EP["ended"] and EP["last_g"] is not None and EP["last_g"] > 0.1:
            lp0 = live_p()
            if lp0 is not None:
                short = INS["prog_end"] - prog_of(lp0)
                SEARCH["shortfall_after_replay_m"] = r4(short)
                need = STALL["hit"] or short > 0.008
        if need:
            u = INS["u"]
            sv = INS["s"]
            qref = INS["quat_end"]
            if qref is None:
                pq, qref = ee_of(get_robot_state())
            p_ref = STALL["p"] if STALL["hit"] else lp0
            prog_ref = prog_of(p_ref)
            target_prog = INS["prog_end"] + 0.005
            STEP = 0.0015
            BACK = 0.008

            def goto(pt, ticks, phase):
                """Short straight-line move of the held peg (IK per tick, gripper closed)."""
                p_from = live_p()
                if p_from is None:
                    return False
                for t in range(1, ticks + 1):
                    if EP["ended"]:
                        return False
                    f = t / float(ticks)
                    tgt = [p_from[j] + (pt[j] - p_from[j]) * f for j in range(3)]
                    ik = solve_ik(Pose(tgt, qref, BASE), seed_joints=EP["last_q"])
                    if not ik.ok:
                        events.append({"ev": "search_ik_fail", "phase": phase, "tick": EP["steps"]})
                        return False
                    send(flist(ik.joint_positions)[:7], 0.0, phase)
                ik = solve_ik(Pose(list(pt), qref, BASE), seed_joints=EP["last_q"])
                if ik.ok:
                    hold(flist(ik.joint_positions)[:7], 0.0, 6, 0.002, phase + "_hold")
                return True

            def push(base, tag):
                """Push along u from base; stop on stall, target depth, or episode end."""
                d0 = prog_of(base)
                cmd_prog = d0
                hist = []
                best = d0
                n = 0
                why = "max_ticks"
                max_n = int((target_prog - d0) / STEP) + 6
                while n < max_n:
                    if EP["ended"]:
                        why = "episode_end"
                        break
                    if EP["steps"] >= MAX_STEPS - 3:
                        why = "budget"
                        break
                    cmd_prog = min(target_prog, cmd_prog + STEP)
                    a = cmd_prog - d0
                    tgt = [base[0] + u[0] * a, base[1] + u[1] * a, base[2]]
                    ik = solve_ik(Pose(tgt, qref, BASE), seed_joints=EP["last_q"])
                    if not ik.ok:
                        why = "ik_fail"
                        break
                    send(flist(ik.joint_positions)[:7], 0.0, "search_push_" + tag)
                    n += 1
                    lp = live_p()
                    if lp is None:
                        why = "no_state"
                        break
                    lpr = prog_of(lp)
                    best = max(best, lpr)
                    hist.append(lpr)
                    if lpr >= target_prog - 0.002:
                        why = "reached_target"
                        break
                    if len(hist) >= 3 and cmd_prog - lpr > 0.003 \
                            and hist[-1] - hist[-3] < 0.0006:
                        why = "stall"
                        break
                return best, why, n

            offsets = [[0.0, 0.0]]
            for k in range(1, 6):
                offsets.append([0.003 * k, 0.0])
                offsets.append([-0.003 * k, 0.0])
            for dz in (0.003, -0.003, 0.006, -0.006):
                offsets.append([0.0, dz])
            for off in offsets:
                if EP["ended"] or EP["steps"] > MAX_STEPS - 25:
                    SEARCH["stopped"] = "budget_or_end"
                    break
                if EP["last_g"] is not None and EP["last_g"] < 0.1:
                    SEARCH["stopped"] = "peg_lost(grip=%.3f)" % EP["last_g"]
                    break
                cur = live_p()
                if cur is None:
                    break
                back_cur = [cur[0] - u[0] * BACK, cur[1] - u[1] * BACK, cur[2]]
                goto(back_cur, 4, "search_back")
                b0 = [p_ref[0] - u[0] * BACK, p_ref[1] - u[1] * BACK, p_ref[2]]
                base = [b0[0] + sv[0] * off[0], b0[1] + sv[1] * off[0], b0[2] + off[1]]
                goto(base, 3, "search_shift")
                start_tick = EP["steps"]
                best, why, n = push(base, "%d_%d" % (int(round(off[0] * 1000)),
                                                     int(round(off[1] * 1000))))
                tr = {"lat_mm": r4(off[0] * 1000), "dz_mm": r4(off[1] * 1000),
                      "start_tick": start_tick, "ticks": n, "best_prog": r4(best),
                      "gain_vs_stall_mm": r4((best - prog_ref) * 1000), "end": why,
                      "grip": r4(EP["last_g"] or 0.0)}
                SEARCH["trials"].append(tr)
                if best > prog_ref + 0.010 or why == "reached_target":
                    SEARCH["inserted"] = True
                    SEARCH["offset_mm"] = [r4(off[0] * 1000), r4(off[1] * 1000)]
                    lp = live_p()
                    if lp is not None and not EP["ended"]:
                        a = target_prog - prog_of(lp)
                        if a > 0.0:
                            # finish seating along the same line
                            best2, why2, n2 = push(lp, "seat")
                            SEARCH["seat"] = {"best_prog": r4(best2), "end": why2, "ticks": n2}
                        hold(list(EP["last_q"]), 0.0, 10, 0.0005, "seat_hold")
                    break
                if EP["ended"]:
                    break
            lpf = live_p() if not EP["ended"] else None
            if lpf is not None:
                SEARCH["final_prog"] = r4(prog_of(lpf))
                SEARCH["final_ee"] = r4list(lpf)
            SEARCH["prog_ref"] = r4(prog_ref)
            SEARCH["target_prog"] = r4(target_prog)
        events.append({"ev": "search_done", "tick": EP["steps"], "needed": need,
                       "inserted": SEARCH["inserted"]})
    except Exception as exc:
        note_error("search", exc)

# ---------------------------------------------------------------- Phase 5: perception fallback
if not use_replay and not EP["ended"]:
    fb = {}
    R["fallback"] = fb
    try:
        peg_set = seg_store.get("peg")
        if peg_set is None or len(peg_set.segmentations) == 0:
            fb["status"] = "no_peg_segmentation"
        else:
            ci = PEG_CHOICE.get("index")
            if ci is None:
                ci = 0
            cand = peg_set.segmentations[ci]
            cloud = mask_to_point_cloud(cand, FRONT, target_frame=BASE)
            pts = points_list(cloud)
            fb["peg_points"] = len(pts)
            if len(pts) >= 10:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                zs = sorted(p[2] for p in pts)
                ztop = zs[int(0.9 * (len(zs) - 1))]
                sxx = sum((p[0] - cx) ** 2 for p in pts)
                syy = sum((p[1] - cy) ** 2 for p in pts)
                sxy = sum((p[0] - cx) * (p[1] - cy) for p in pts)
                peg_yaw = 0.5 * math.atan2(2 * sxy, sxx - syy)
                fb["peg_center"] = [r4(cx), r4(cy), r4(ztop)]
                fb["peg_yaw"] = r4(peg_yaw)
                rs = get_robot_state()
                p_home, q_home = ee_of(rs)
                Rh = quat_to_mat(q_home)
                yaw_y = math.atan2(Rh[1][1], Rh[0][1])  # EE y-axis (finger closing axis)
                want = peg_yaw + math.pi / 2.0
                dth = want - yaw_y
                while dth > math.pi / 2:
                    dth -= math.pi
                while dth < -math.pi / 2:
                    dth += math.pi
                qz = [math.cos(dth / 2), 0.0, 0.0, math.sin(dth / 2)]
                q_grasp = quat_mul(qz, q_home)
                gz = ztop - 0.01
                open_gripper()
                r1 = move_to_pose(Pose([cx, cy, gz + 0.10], q_grasp, BASE), tolerance=0.01,
                                  max_steps=150)
                r2 = move_to_pose(Pose([cx, cy, gz], q_grasp, BASE), tolerance=0.008,
                                  max_steps=100)
                r3 = close_gripper()
                rs = get_robot_state()
                fb["grip_after_close"] = r4(rs.gripper_positions["primary"])
                r4m = move_to_pose(Pose([cx, cy, gz + 0.12], q_grasp, BASE), tolerance=0.01,
                                   max_steps=150)
                fb["moves_ok"] = [bool(r1.ok), bool(r2.ok), bool(r3.ok), bool(r4m.ok)]
                fb["status"] = "lifted"
    except Exception as exc:
        note_error("fallback", exc)

# ---------------------------------------------------------------- Phase 6: wrap-up
R["episode"] = {"ticks_sent": EP["steps"], "ended": EP["ended"], "end_reason": EP["end_reason"],
                "terminated": EP["term"], "truncated": EP["trunc"], "flags_seen": EP["flags_seen"]}
if not EP["ended"]:
    try:
        rs = get_robot_state()
        p, qq = ee_of(rs)
        fin = {"ee_pos": r4list(p), "gripper": r4(rs.gripper_positions["primary"]),
               "timestamp_s": r4(rs.timestamp_s)}
        if valid_idx and vee[valid_idx[-1]] is not None:
            fin["video_final_ee"] = r4list(vee[valid_idx[-1]])
            fin["ee_delta_to_video_final_m"] = r4(dist3(p, vee[valid_idx[-1]]))
        R["final_state"] = fin
    except Exception as exc:
        note_error("final_state", exc)
