"""RouteStick controller: reproduce the stick path shown in the conditioning video.

The conditioning-video manifest records the robot's end-effector state for each
retained video frame. We rebuild that path (position over source-frame time) and
track it closed-loop with official delta rows.
"""
import json, time, os, math, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path

WS = Path("/workspace")
OUT = WS / "output"
LOG = None
STEP = 0.05  # m per unit action
W_IN, W_OUT = 0.022, 0.032  # full tip lowering within W_IN of a target, none beyond W_OUT
MAX_DZ_STEP = 0.01  # max route z change per control step (densify otherwise)
PULL, PULL_R = 0.7, 0.04  # attraction of the path to target centres
Z_MIN = 0.015
ALPHA0 = 0.4  # measured: one step realises ~40% of the commanded delta


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    try:
        with open(OUT / "controller_log.txt", "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def packet(result):
    for b in result.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}


def load_manifest(obs):
    seq = obs.get("frame_sequence") or {}
    mp = seq.get("manifest_path") or os.environ.get("RS_MANIFEST")
    if not mp:
        return None
    return json.loads((WS / mp).read_text())


def layout_mapper(man):
    """Detect target discs in the first video frame and the live frame; build a mapper."""
    from perception import frame_targets, line_inliers, order_along_line, RouteMapper
    vids = [f for f in man["frames"] if f.get("phase") == "video_demo"]
    exe = [f for f in man["frames"] if f.get("phase") == "execution"]
    if not vids or not exe:
        return None, "no frames"
    src = order_along_line(line_inliers(frame_targets(vids[0], WS)))
    dst = order_along_line(line_inliers(frame_targets(exe[-1], WS)))
    info = {"src": [[round(c, 3) for c in t["xyz"]] for t in src], "dst": [[round(c, 3) for c in t["xyz"]] for t in dst]}
    if len(src) < 2 or len(src) != len(dst):
        return None, info
    return RouteMapper([t["xyz"] for t in src], [t["xyz"] for t in dst]), info


def load_route(man):
    pts = []
    for fr in man["frames"]:
        if fr.get("phase") != "video_demo":
            continue
        st = fr.get("state")
        if not st or len(st) < 3:
            continue
        pts.append((fr.get("source_frame_index", fr.get("frame_index")), st[:3]))
    pts.sort(key=lambda p: p[0])
    return pts


def interp_route(pts):
    """Return per-source-frame positions (linear interpolation over gaps)."""
    out = []
    for (i0, p0), (i1, p1) in zip(pts[:-1], pts[1:]):
        n = max(1, i1 - i0)
        for k in range(n):
            t = k / n
            out.append([p0[j] + (p1[j] - p0[j]) * t for j in range(3)])
    out.append(pts[-1][1])
    return out


def rot_correction(cur_aa, ref_R, gain=0.5):
    """World-frame rotation vector (scaled by gain) taking cur orientation toward ref_R."""
    import numpy as np
    R = ref_R @ aa_mat(cur_aa).T
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2)); th = math.acos(c)
    if th < 1e-6:
        return [0.0, 0.0, 0.0]
    if th > math.pi - 1e-3:
        return [0.0, 0.0, 0.0]
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return [float(x) for x in w * th * gain]


def aa_mat(v):
    import numpy as np
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def _main(ctx):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    obs = packet(ctx.call("robomme_observe"))
    log("instruction:", obs.get("instruction"), "step", obs.get("step_id"))
    man = load_manifest(obs)
    pts = load_route(man) if man else None
    if not pts:
        log("no video route; abort")
        return
    route = interp_route(pts)
    try:
        mapper, info = layout_mapper(man)
        log("targets", json.dumps(info))
    except Exception as e:  # perception failure -> identity mapping
        mapper = None
        log("layout mapping failed", repr(e))
    if mapper is not None:
        route = [mapper.map(p) for p in route]
        log("route mapped to live layout")
    else:
        log("WARNING: identity route mapping")
    # tool geometry: the live stick can differ from the video's; follow the video tip path
    live_targets = []
    try:
        from perception import frame_targets, line_inliers, order_along_line
        exe = [f for f in man["frames"] if f.get("phase") == "execution"]
        live_targets = [t["xyz"][:2] for t in order_along_line(line_inliers(frame_targets(exe[-1], WS)))]
    except Exception as e:
        log("live target detection failed", repr(e))
    dz, o_v, o_l = 0.0, [0, 0, 0], [0, 0, 0]
    try:
        from perception import stick_tip_offset
        vids = [f for f in man["frames"] if f.get("phase") == "video_demo"]
        ov = stick_tip_offset(vids[0], WS); ol = stick_tip_offset(exe[-1], WS)
        log("tip offsets video", ov, "live", ol)
        if ov and ol:
            o_v, o_l = ov[0], ol[0]
            dz = max(-0.12, min(0.12, o_v[2] - o_l[2]))
    except Exception as e:
        log("tip offset failed", repr(e))
    # hand yaw: when the stick is shorter the hand comes low; turn its long axis
    # (initially along world y) perpendicular to the target line so it fits between blocks
    yaw = 0.0
    if dz < -0.01 and len(live_targets) >= 2:
        d = [live_targets[-1][0] - live_targets[0][0], live_targets[-1][1] - live_targets[0][1]]
        ang = math.atan2(d[1], d[0])  # line direction; hand axis wanted at ang+90deg; hand axis now at +90deg
        yaw = (ang + math.pi / 2) % math.pi - math.pi / 2
    cy, sy = math.cos(yaw), math.sin(yaw)
    o_l_rot = [cy * o_l[0] - sy * o_l[1], sy * o_l[0] + cy * o_l[1]]
    cxy = [max(-0.02, min(0.02, o_v[0] - o_l_rot[0])), max(-0.02, min(0.02, o_v[1] - o_l_rot[1]))]
    log("dz", round(dz, 4), "yaw", round(yaw, 3), "xy corr", [round(c, 4) for c in cxy], "live targets", [[round(c, 3) for c in t] for t in live_targets])

    def weight(p):
        if dz >= 0 or not live_targets:
            return 1.0
        dmin = min(math.hypot(p[0] - t[0], p[1] - t[1]) for t in live_targets)
        return max(0.0, min(1.0, (W_OUT - dmin) / (W_OUT - W_IN)))

    # tip lowering only inside a small zone around each target (the hand is then
    # clear of the blocks); the route is densified where z changes fast so the arm
    # reaches the lowered height before passing the target.
    newroute = []
    for k, p in enumerate(route):
        w = weight(p)
        if dz >= 0:
            w = 1.0
        else:
            w *= max(0.0, min(1.0, (0.16 - p[2]) / 0.06))
        q = [p[0], p[1]]
        if live_targets and p[2] < 0.12:
            # pull the path toward a nearby target centre for a safer visit margin
            t = min(live_targets, key=lambda t: math.hypot(p[0] - t[0], p[1] - t[1]))
            dd = math.hypot(p[0] - t[0], p[1] - t[1])
            f = PULL * max(0.0, 1.0 - dd / PULL_R)
            q = [p[0] + (t[0] - p[0]) * f, p[1] + (t[1] - p[1]) * f]
        newroute.append([q[0] + cxy[0], q[1] + cxy[1], max(Z_MIN, p[2] + dz * w)])
    dense = [newroute[0]]
    for p in newroute[1:]:
        q = dense[-1]
        nsub = max(1, int(math.ceil(abs(p[2] - q[2]) / MAX_DZ_STEP)))
        for m in range(1, nsub + 1):
            t = m / nsub
            dense.append([q[a] + (p[a] - q[a]) * t for a in range(3)])
    log("route densified", len(newroute), "->", len(dense))
    newroute = dense
    route = newroute
    ref_aa = aa_mat([0, 0, yaw]) @ aa_mat(obs["state"][3:6])
    limit = obs.get("episode_limits", {}).get("remaining_episode_steps", 1000)
    log("route frames", len(route), "remaining steps", limit)
    state = obs["state"]
    i = int(os.environ.get("RS_START", "0"))
    CH = int(os.environ.get("RS_CHUNK", "5"))
    alpha = ALPHA0
    ended = False
    maxerr = 0.0
    res = {}
    # turn the hand to the planned yaw in place (high above the table)
    for _ in range(40):
        rc = rot_correction(state[3:6], ref_aa, 1.0 / ALPHA0)
        if math.sqrt(sum(x * x for x in rc)) * ALPHA0 < 0.02:
            break
        row = [0, 0, 0] + [clip(r / 0.5) for r in rc] + [-1.0]
        res = packet(ctx.call("robomme_step", actions=[row]))
        state = res.get("state", state)
    log("yaw phase done at step", res.get("step_id"), "aa", [round(x, 3) for x in state[3:6]])
    # approach the starting route point if not there
    for _ in range(20):
        e = [route[i][a] - state[a] for a in range(3)]
        if math.sqrt(sum(x * x for x in e)) < 0.003:
            break
        row = [clip(e[a] / (STEP * alpha)) for a in range(3)] + [0, 0, 0, -1.0]
        res = packet(ctx.call("robomme_step", actions=[row]))
        state = res.get("state", state)
    errs = []
    while i < len(route) - 1 and not ended:
        rem = (res.get("episode_limits") or obs.get("episode_limits", {})).get("remaining_episode_steps", 10**6)
        j = min(len(route) - 1, i + CH, i + max(1, rem - 2))
        rows = []
        cur = list(state[:3])
        rc = rot_correction(state[3:6], ref_aa, 0.3)
        for k in range(i + 1, j + 1):
            tgt = route[k]
            d = [(tgt[a] - cur[a]) / (STEP * alpha) for a in range(3)]
            r = rc if k == i + 1 else [0, 0, 0]
            rows.append([clip(d[0]), clip(d[1]), clip(d[2]), clip(r[0] / 0.5), clip(r[1] / 0.5), clip(r[2] / 0.5), -1.0])
            cur = tgt
        p_before = list(state[:3])
        res = packet(ctx.call("robomme_step", actions=rows))
        state = res.get("state", state)
        # online estimate of the per-step realisation ratio alpha
        cmd = [sum(r[a] for r in rows) * STEP for a in range(3)]
        disp = [state[a] - p_before[a] for a in range(3)]
        cc = sum(c * c for c in cmd)
        if cc > 0.01 ** 2 and len(rows) >= 3:
            ah = sum(c * d for c, d in zip(cmd, disp)) / cc
            if 0.15 < ah < 1.2:
                alpha = min(0.9, max(0.25, 0.7 * alpha + 0.3 * ah))
        err = math.dist(state[:3], route[j])
        errs.append(err)
        maxerr = max(maxerr, err)
        if res.get("episode_ended"):
            ended = True
            log("episode ended", res.get("transition"))
        rem = res.get("episode_limits", {}).get("remaining_episode_steps")
        if i % 50 < CH:
            log(f"i={j} step={res.get('step_id')} rem={rem} pos={[round(x,3) for x in state[:3]]} tgt={[round(x,3) for x in route[j]]} err={err:.4f} alpha={alpha:.3f}")
        i = j
        if rem is not None and rem <= 1:
            break
    log("route done; maxerr", round(maxerr, 4), "elapsed", round(time.time() - t0, 1))
    if not ended:
        # settle at the final point
        for _ in range(3):
            tgt = route[-1]
            d = [(tgt[a] - state[a]) / (STEP * ALPHA0) for a in range(3)]
            res = packet(ctx.call("robomme_step", actions=[[clip(d[0]), clip(d[1]), clip(d[2]), 0, 0, 0, -1.0]] * 2))
            state = res.get("state", state)
            if res.get("episode_ended"):
                log("episode ended", res.get("transition")); break
        log("final pos", [round(x, 3) for x in state[:3]], "ended", res.get("episode_ended"))


def main(ctx):
    try:
        _main(ctx)
    except Exception as e:
        import traceback
        log("controller exception", repr(e), traceback.format_exc())
