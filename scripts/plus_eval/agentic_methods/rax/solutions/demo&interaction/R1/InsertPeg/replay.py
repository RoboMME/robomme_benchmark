"""Primary method: track this episode's own conditioning-video TCP trajectory.

The conditioning video of InsertPeg shows the robot doing exactly the required
task (same peg, same end, same side) in the same layout as the execution start
(verified offline on all 3 training recordings: execution frame 0 equals video
frame 0). If the reset manifest carries the robot state for each video frame, the
video TCP poses and finger widths are a complete plan for this episode. We track
them closed-loop (target = video pose, delta from the measured pose each step).
"""
import json
import os

import numpy as np

import geom
from rio import log, WS


def load_manifest(pkt):
    paths = []
    for p, v in _find_manifest_paths(pkt):
        paths.append(v)
    if not paths:
        # inline frame list fallback
        from rio import find_all
        for p, v in find_all(pkt, "frames"):
            if isinstance(v, list) and v and isinstance(v[0], dict) and "phase" in v[0]:
                return {"frames": v}, "inline:" + "/".join(str(x) for x in p)
    for mp in paths:
        for base in (WS, ""):
            full = os.path.join(base, mp) if base else mp
            if os.path.isfile(full):
                try:
                    with open(full) as f:
                        m = json.load(f)
                    return m, full
                except Exception as e:
                    log("manifest read failed", full, repr(e))
    return None, None


def _find_manifest_paths(pkt):
    from rio import find_all
    out = []
    for p, v in find_all(pkt, "manifest_path"):
        if isinstance(v, str):
            out.append((p, v))
    for p, v in find_all(pkt, "frame_sequence"):
        if isinstance(v, str) and v.endswith(".json"):
            out.append((p, v))
    return out


def split_frames(manifest):
    frames = manifest.get("frames", []) if isinstance(manifest, dict) else []
    def key(f):
        fi = f.get("frame_index")
        return fi if isinstance(fi, (int, float)) else 1e9
    frames = sorted(frames, key=key)
    video = [f for f in frames if f.get("phase") == "video_demo"]
    execf = [f for f in frames if f.get("phase") == "execution"]
    return video, execf


def file_of(frame, camera, kind):
    for f in frame.get("files", []) or []:
        if f.get("camera") == camera and f.get("kind") == kind:
            return os.path.join(WS, f.get("path", "")), f
    return None, None


def video_states(video):
    out = []
    for f in video:
        s = f.get("state")
        if isinstance(s, (list, tuple)) and len(s) == 8:
            try:
                a = np.array(s, dtype=float)
                if np.all(np.isfinite(a)):
                    t = f.get("timestamp_s")
                    out.append((float(t) if isinstance(t, (int, float)) else None, a))
            except Exception:
                pass
    return out


def states_usable(vs, exec_state):
    if len(vs) < 30:
        return False, "too few video states (%d)" % len(vs)
    P = np.array([s[:3] for _, s in vs])
    rng = float(np.max(np.ptp(P, axis=0)))
    if rng < 0.03:
        return False, "video states do not move (range %.4f)" % rng
    if exec_state is not None:
        d = float(np.linalg.norm(P[0] - exec_state[:3]))
        if d > 0.06:
            return False, "video start %.3f m from execution start" % d
    W = np.array([s[6] + s[7] for _, s in vs])
    if float(np.ptp(W)) < 0.004:
        return False, "video gripper never moves (ptp %.4f)" % float(np.ptp(W))
    return True, "ok range=%.3f wptp=%.3f" % (rng, float(np.ptp(W)))


def build_waypoints(vs):
    """Per-control-step waypoints (pos, R, width), interpolating if the video was subsampled."""
    wps = []
    for i, (t, s) in enumerate(vs):
        if i > 0:
            t0, s0 = vs[i - 1]
            n = 1
            if t is not None and t0 is not None:
                n = int(round((t - t0) / 0.05))
                n = max(1, min(n, 20))
            if n > 1:
                R0 = geom.rotvec_to_mat(s0[3:6])
                R1 = geom.rotvec_to_mat(s[3:6])
                dr = geom.mat_to_rotvec(R1 @ R0.T)
                for k in range(1, n):
                    a = k / n
                    wps.append((s0[:3] * (1 - a) + s[:3] * a,
                                geom.rotvec_to_mat(dr * a) @ R0,
                                float((s0[6] + s0[7]) * (1 - a) + (s[6] + s[7]) * a)))
        wps.append((s[:3].copy(), geom.rotvec_to_mat(s[3:6]), float(s[6] + s[7])))
    return wps


def gripper_commands(widths, init_cmd=-1.0, thr=0.005, look=10):
    """Per-waypoint gripper command from substantial, sustained finger-width changes."""
    W = np.asarray(widths, dtype=float)
    n = len(W)
    cmd = np.zeros(n)
    c = init_cmd
    for i in range(n):
        j = min(n - 1, i + look)
        seg = W[i:j + 1]
        if c < 0 and W[i] - seg.min() > thr and W[j] < W[i] - thr * 0.8:
            # closing begins where the width starts to fall
            c = 1.0
        elif c > 0 and seg.max() - W[i] > thr and W[j] > W[i] + thr * 0.8:
            c = -1.0
        cmd[i] = c
    return cmd


def run_replay(robot, vs):
    wps = build_waypoints(vs)
    widths = [w for _, _, w in wps]
    cmds = gripper_commands(widths, init_cmd=-1.0 if robot.width() > 0.03 else 1.0)
    N = len(wps)
    changes = [i for i in range(1, N) if cmds[i] != cmds[i - 1]]
    log("replay: %d waypoints, gripper changes at %s, width range %.4f..%.4f" % (
        N, changes, min(widths), max(widths)))
    # step budget: keep margin for holds and the final push
    stride = 1
    if robot.remaining is not None and robot.remaining >= 100:
        need = N + 25 * max(1, len(changes)) + 60
        if need > robot.remaining:
            stride = int(np.ceil(N / max(1.0, robot.remaining - 25 * max(1, len(changes)) - 60)))
            stride = max(1, min(stride, 4))
            log("replay: step budget tight (remaining %s, need %d) -> stride %d" % (
                robot.remaining, need, stride))
    protect = set()
    for c in changes:
        protect.update(range(max(0, c - 20), min(N, c + 20)))
    protect.update(range(max(0, N - 80), N))

    cur_cmd = cmds[0]
    i = 1
    max_err = 0.0
    errs = []
    grasp_checked = False
    catchup_budget = 150
    while i < N and robot.can_move():
        p_t, R_t, w_t = wps[i]
        if cmds[i] != cur_cmd:
            # gripper transition: go to the pose where the video's fingers finished moving,
            # then settle the fingers there
            new = cmds[i]
            j_end = i
            for j in range(i, min(N - 3, i + 40)):
                if abs(widths[j] - widths[i]) > 0.002 and abs(widths[j + 3] - widths[j]) < 0.0005:
                    j_end = j
                    break
            else:
                j_end = min(N - 1, i + 10)
            # onset of the finger motion in the video
            k0 = i
            for k in range(i, j_end + 1):
                if abs(widths[k] - widths[i]) > 0.001:
                    k0 = k
                    break
            # do not chase a pose the arm moved far away to while the fingers moved
            if np.linalg.norm(wps[j_end][0] - wps[k0][0]) > 0.01:
                j_end = k0
            pg, Rg, _ = wps[j_end]
            robot.goto(pg, Rg, cur_cmd, tol=0.0025, rtol=0.03, max_steps=20, max_pos=0.3,
                       label="pre-grip %d" % j_end)
            ws = robot.grip_settle(pg, Rg, new)
            log("grip %+.0f at wp %d..%d: widths %s (video after: %.4f) pos_err=%.4f" % (
                new, i, j_end, ["%.4f" % w for w in ws[-4:]], widths[min(N - 1, j_end + 5)],
                float(np.linalg.norm(pg - robot.pos()))))
            cur_cmd = new
            i = j_end + 1
            if new > 0 and not grasp_checked:
                grasp_checked = True
                check_grasp(robot, wps, widths, i, pg, Rg)
            continue
        row = robot.action_to(p_t, R_t, cur_cmd, max_pos=1.0, max_rot=1.0)
        prev_rerr = geom.rot_angle(R_t, robot.R())
        # adaptive chunking when tool calls are slow (time budget), never near critical phases
        k = chunk_size(robot, N - i)
        if k > 1 and i not in protect:
            rows = [row]
            j = i
            while len(rows) < k and j + stride < N and cmds[j + stride] == cur_cmd and (j + stride) not in protect:
                a, b = wps[j], wps[j + stride]
                dp = (b[0] - a[0]) / 0.05
                if robot.rot_mode == "world":
                    dr = geom.mat_to_rotvec(b[1] @ a[1].T) / 0.5
                else:
                    dr = geom.mat_to_rotvec(a[1].T @ b[1]) / 0.5
                rows.append([*np.clip(dp, -1, 1), *np.clip(dr, -1, 1), cur_cmd])
                j += stride
            robot.step_rows(rows)
            i = j + stride
            if robot.state is None:
                log("replay: lost state")
                return False
            continue
        robot.step_rows([row])
        if robot.state is None:
            log("replay: lost state")
            return False
        e = float(np.linalg.norm(p_t - robot.pos()))
        er = geom.rot_angle(R_t, robot.R())
        robot._check_rot_convention(prev_rerr, er)
        errs.append(e)
        max_err = max(max_err, e)
        if i % 25 == 0:
            log("wp %d/%d err=%.4f rerr=%.3f width=%.4f cmd=%+.0f rem=%s" % (
                i, N, e, er, robot.width(), cur_cmd, robot.remaining))
        # catch-up when lagging (bounded)
        tries = 0
        while (e > 0.012 or er > 0.08) and tries < 3 and catchup_budget > 0 and robot.can_move():
            catchup_budget -= 1
            row = robot.action_to(p_t, R_t, cur_cmd, max_pos=1.0, max_rot=1.0)
            robot.step_rows([row])
            e = float(np.linalg.norm(p_t - robot.pos()))
            er_prev = er
            er = geom.rot_angle(R_t, robot.R())
            robot._check_rot_convention(er_prev, er)
            tries += 1
        if i in protect:
            i += 1
        else:
            i += stride
    log("replay done: max_err=%.4f mean_err=%.4f complete=%s ended=%s final_cmd=%+.0f" % (
        max_err, float(np.mean(errs)) if errs else -1, robot.complete, robot.ended, cur_cmd))
    robot.final_grip_cmd = float(cur_cmd)
    return True


def chunk_size(robot, left):
    """Rows per call so that the remaining waypoints fit in the controller's time budget."""
    if robot.step_calls < 20:
        return 1
    from rio import elapsed
    per_call = robot.step_time / max(1, robot.step_calls)
    t_left = robot.deadline - elapsed() - 150.0  # keep time for the final push
    if t_left <= 0:
        return 5
    need = left * per_call
    if need <= t_left:
        return 1
    return int(min(5, np.ceil(need / t_left)))


def check_grasp(robot, wps, widths, i, p_t, R_t):
    """After closing: compare the finger width with the video's; one regrasp attempt if empty."""
    vid_w = widths[min(len(widths) - 1, i + 10)]
    w = robot.width()
    log("grasp check: width %.4f video %.4f" % (w, vid_w))
    if w > 0.003 or vid_w < 0.003:
        return True
    # empty grasp: open, rise, re-descend with tight convergence, close
    log("grasp appears empty -> one regrasp")
    up = p_t + np.array([0, 0, 0.05])
    robot.goto(up, R_t, -1.0, tol=0.004, max_steps=30, label="regrasp-up")
    robot.hold(-1.0, 8)
    robot.goto(p_t, R_t, -1.0, tol=0.002, max_steps=40, max_pos=0.2, label="regrasp-down")
    ws = robot.grip_settle(p_t, R_t, 1.0)
    log("regrasp widths", ["%.4f" % x for x in ws[-3:]])
    return robot.width() > 0.003


def final_push(robot, wps):
    """After the replay: if not complete, push a little further along the insertion direction."""
    if not robot.can_move():
        return
    grip = getattr(robot, "final_grip_cmd", 1.0)
    if grip < 0:
        log("final push skipped: the video ends with the gripper open")
        return
    p_f, R_f, _ = wps[-1]
    d = None
    for k in range(5, min(len(wps), 120)):
        v = p_f - wps[-1 - k][0]
        v[2] = 0.0
        if np.linalg.norm(v) > 0.015:
            d = v / np.linalg.norm(v)
            break
    log("final push: direction", None if d is None else np.round(d, 3))
    robot.goto(p_f, R_f, 1.0, tol=0.002, rtol=0.02, max_steps=25, max_pos=0.15, label="final-converge")
    if d is None or not robot.can_move():
        return
    # perpendicular horizontal direction for small wiggles
    perp = np.array([-d[1], d[0], 0.0])
    z = np.array([0, 0, 1.0])
    # rung 1: push straight in
    for depth in (0.006, 0.012, 0.018):
        if not robot.can_move():
            return
        robot.goto(p_f + d * depth, R_f, 1.0, tol=0.002, rtol=0.02, max_steps=10, max_pos=0.1,
                   label="push %.3f" % depth)
        adv = float(np.dot(robot.pos() - p_f, d))
        log("push depth %.3f advanced %.4f complete=%s" % (depth, adv, robot.complete))
        if robot.complete or robot.ended:
            return
        if adv < depth - 0.004:
            break  # blocked
    # rung 2: back off and retry from small lateral/vertical offsets (spiral, <= 6 mm)
    offsets = []
    for r in (0.002, 0.004, 0.006):
        for (a, b) in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            offsets.append((a * r, b * r))
    for (a, b) in offsets:
        if not robot.can_move():
            return
        back = p_f - d * 0.015 + perp * a + z * b
        robot.goto(back, R_f, 1.0, tol=0.003, rtol=0.03, max_steps=10, max_pos=0.15, label="spiral-back")
        robot.goto(p_f + d * 0.012 + perp * a + z * b, R_f, 1.0, tol=0.002, rtol=0.02, max_steps=14,
                   max_pos=0.1, label="spiral-push")
        adv = float(np.dot(robot.pos() - p_f, d))
        if robot.complete or robot.ended:
            log("spiral: complete at offset", (a, b))
            return
        if adv > 0.008:
            log("spiral: advanced %.4f at offset %s" % (adv, (a, b)))
    log("final push: no success")
