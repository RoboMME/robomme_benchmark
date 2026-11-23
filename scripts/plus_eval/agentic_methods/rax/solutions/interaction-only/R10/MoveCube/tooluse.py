"""Generic tool use: grasp the hook or peg, align, lower, move the cube to the target."""
import math, time
import numpy as np
from motion import move_to, hold, wrap, log
from perc import world_points
from scene import locate

Z_SAFE = 0.14


def pink_geometry(meta, imgs, tool):
    rgb = imgs[("front", "rgb")].astype(int)
    P, valid = world_points(meta, "front", imgs[("front", "depth")])
    a = tool["tool_axis"]; n = np.array([-a[1], a[0]]); h = tool["handle"][:2]
    L = tool["tool_len"] if 0.08 < tool["tool_len"] < 0.36 else 0.30
    rel = P[..., :2] - h; s = rel @ a; l = rel @ n
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    pink = (r > 130) & (r - g > 35) & (r - g < 90) & (g - b < 32) & (P[..., 2] > 0.005) & (P[..., 2] < 0.05)
    m = pink & (s > L - 0.06) & (s < L + 0.08) & (np.abs(l) < 0.12)
    geo = {"s_tip": L, "l_mean": 0.0, "side": 1.0, "spread": 0.0, "pink_px": int(m.sum()), "L": L}
    if m.sum() >= 6:
        lm = l[m]
        geo["side"] = 1.0 if np.median(lm) > 0 else -1.0
        geo["spread"] = float(np.percentile(np.abs(lm), 95))
        geo["s_tip"] = float(np.median(s[m]))
        geo["l_mean"] = float(np.mean(lm))
    if abs(geo["l_mean"]) > 0.025:
        geo["kind"] = "hook"
    elif geo["spread"] > 0.022:
        geo["kind"] = "scoop"
    else:
        geo["kind"] = "peg"
    if m.sum() < 6:
        geo["kind"] = "unknown"
    return geo


def obstacle_points(meta, imgs, exclude):
    """Elevated points from the front view (xy), minus excluded discs [(xy, r)]."""
    P, valid = world_points(meta, "front", imgs[("front", "depth")])
    m = valid & (P[..., 2] > 0.012) & (P[..., 2] < 0.3) & (np.abs(P[..., 0]) < 0.8) & (np.abs(P[..., 1]) < 0.8)
    m &= ~((P[..., 0] < -0.5) & (np.abs(P[..., 1]) < 0.15))  # robot base
    pts = P[m][:, :2]
    keep = np.ones(len(pts), bool)
    for xy, r in exclude:
        keep &= np.linalg.norm(pts - np.asarray(xy)[:2], axis=1) > r
    return pts[keep][::2]


def seg_cost(pts, A, B, r):
    if len(pts) == 0: return 0
    AB = B - A; L2 = max(AB @ AB, 1e-9)
    t = np.clip((pts - A) @ AB / L2, 0, 1)
    dd = np.linalg.norm(pts - (A + t[:, None] * AB), axis=1)
    return int((dd < r).sum())


def grasp_tool(rb, tool):
    a = tool["tool_axis"]; alpha = math.atan2(a[1], a[0])
    gy = wrap(alpha)
    if gy > math.pi / 2: gy -= math.pi
    if gy < -math.pi / 2: gy += math.pi
    h = tool["handle"]; z = max(0.012, tool["handle_top"] - 0.008)
    move_to(rb, [rb.pos()[0], rb.pos()[1], Z_SAFE], None, -1, max_steps=40)
    move_to(rb, [h[0], h[1], Z_SAFE], gy, -1, max_steps=80)
    move_to(rb, [h[0], h[1], z + 0.03], gy, -1, max_steps=40)
    move_to(rb, [h[0], h[1], z], gy, -1, max_steps=40, vmax=0.4)
    hold(rb, 8, 1.0)
    tcp = rb.pos()
    n = np.array([-a[1], a[0]])
    model = {"off": wrap(alpha - rb.yaw()), "z_grasp": float(tcp[2]),
             "tcp_s": float((tcp[:2] - h[:2]) @ a), "tcp_l": float((tcp[:2] - h[:2]) @ n),
             "w": rb.grip_open()}
    move_to(rb, [tcp[0], tcp[1], Z_SAFE], None, 1.0, max_steps=50)
    model["w_lifted"] = rb.grip_open()
    log("grasp", model)
    return model


def plan(model, geo, kind, cube, target, obst, rel_ang, gap, lat):
    D = target[:2] - cube[:2]; dist = float(np.linalg.norm(D)); p = D / max(dist, 1e-6)
    if kind == "hook":
        u = -p  # tool axis points away from target; bar behind the cube
        nn = np.array([-u[1], u[0]]) * geo["side"]
        s_c = geo["s_tip"] - 0.03 - gap
        hc = cube[:2] - u * s_c - nn * lat
    else:
        ca, sa = math.cos(rel_ang), math.sin(rel_ang)
        u = np.array([ca * p[0] - sa * p[1], sa * p[0] + ca * p[1]])
        head_r = 0.032 if kind == "scoop" else 0.015
        tip = cube[:2] - p * (0.02 + head_r + gap)
        hc = tip - u * geo["s_tip"]
    a = u; n = np.array([-a[1], a[0]])
    tcp = hc + a * model["tcp_s"] + n * model["tcp_l"]
    yaw = wrap(math.atan2(u[1], u[0]) - model["off"])
    move = p * (dist + gap)
    pink_goal = hc + a * geo["s_tip"] + n * geo["l_mean"]
    L = geo["L"] + 0.02
    # footprint: tool segment at start and end, gripper disc
    c = 0
    for k in (0.0, 0.5, 1.0):
        A = tcp + move * k; B = A + u * (L - model["tcp_s"])
        c += seg_cost(obst, A, B, 0.025)
        c += seg_cost(obst, A, A, 0.06)
    base = np.array([-0.615, 0.0])
    reach = max(np.linalg.norm(tcp - base), np.linalg.norm(tcp + move - base))
    near = min(np.linalg.norm(tcp - base), np.linalg.norm(tcp + move - base))
    pen = c + (1000 if reach > 0.80 else 0) + (1000 if near < 0.25 else 0) + abs(rel_ang) * 60
    return {"tcp": tcp, "yaw": yaw, "move": move, "pink_goal": pink_goal, "dist": dist, "cost": pen, "rel": rel_ang, "u": u}


def measure_pink(rb, goal, scene_obs):
    try:
        meta, imgs = rb.observe()
        rgb = imgs[("front", "rgb")].astype(int)
        P, valid = world_points(meta, "front", imgs[("front", "depth")])
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        pink = (r > 130) & (r - g > 35) & (r - g < 90) & (g - b < 32) & valid
        pink &= (P[..., 2] > 0.02) & (P[..., 2] < 0.12)
        pink &= np.linalg.norm(P[..., :2] - goal[:2], axis=-1) < 0.09
        if pink.sum() < 5: return None
        return np.mean(P[pink][:, :2], 0)
    except Exception as ex:
        log("measure_pink exc", repr(ex)); return None


def execute(rb, model, pl, scene_obs=True):
    tcp, yaw = pl["tcp"], pl["yaw"]
    move_to(rb, [rb.pos()[0], rb.pos()[1], Z_SAFE], None, 1.0, max_steps=40)
    move_to(rb, [rb.pos()[0], rb.pos()[1], Z_SAFE], yaw, 1.0, max_steps=60)
    move_to(rb, [tcp[0], tcp[1], Z_SAFE], yaw, 1.0, max_steps=80)
    zt = model["z_grasp"] + 0.004
    move_to(rb, [tcp[0], tcp[1], zt + 0.04], yaw, 1.0, max_steps=40)
    if scene_obs is not None:
        for it in range(3):
            meas = measure_pink(rb, pl["pink_goal"], scene_obs)
            if meas is None:
                log("align: pink not found"); break
            e = pl["pink_goal"] - meas
            log("align it", it, "pink err", np.round(e, 4))
            if np.linalg.norm(e) > 0.08: break
            if np.linalg.norm(e) < 0.004: break
            tcp = rb.pos()[:2] + e
            move_to(rb, [tcp[0], tcp[1], zt + 0.04], yaw, 1.0, max_steps=20, vmax=0.3)
    move_to(rb, [tcp[0], tcp[1], zt], yaw, 1.0, max_steps=40, vmax=0.3)
    zr = rb.pos()[2]
    log("lowered z", round(zr, 4), "target", round(zt, 4), "xy err", np.round(rb.pos()[:2] - tcp, 4))
    if zr > zt + 0.012:
        move_to(rb, [rb.pos()[0], rb.pos()[1], Z_SAFE], None, 1.0, max_steps=40)
        return "blocked"
    p0 = rb.pos(); goal = p0.copy(); goal[:2] += pl["move"]; goal[2] = zt
    nseg = max(2, int(math.ceil(np.linalg.norm(pl["move"]) / 0.03)))
    for k in range(1, nseg + 1):
        move_to(rb, p0 + (goal - p0) * k / nseg, yaw, 1.0, max_steps=25, vmax=0.3)
        if rb.ended: return "ended"
    hold(rb, 3, 1.0)
    p = rb.pos()
    # retreat slightly backwards before lifting so the tool does not drag the cube
    back = -pl["move"] / max(np.linalg.norm(pl["move"]), 1e-6) * 0.015
    move_to(rb, [p[0] + back[0], p[1] + back[1], p[2] + 0.003], yaw, 1.0, max_steps=10, vmax=0.3)
    move_to(rb, [rb.pos()[0], rb.pos()[1], Z_SAFE], yaw, 1.0, max_steps=40)
    return "moved"


def run_tool(rb, want, t_end, scene_fn):
    meta, imgs, sc = scene_fn(rb)
    tools = sc.get("tools", [])
    if not tools or "cube" not in sc or "target" not in sc:
        log("tool: missing", list(sc)); return False
    geos = [pink_geometry(meta, imgs, t) for t in tools]
    for t, g in zip(tools, geos): log("tool", np.round(t["handle"], 3), g)
    idx = [i for i, g in enumerate(geos) if g["kind"] == want] or \
          [i for i, g in enumerate(geos) if g["kind"] == "unknown"]
    if not idx and want == "scoop":
        idx = [int(np.argmin([t["tool_len"] for t in tools]))]
    i = idx[0] if idx else 0
    tool, geo = tools[i], geos[i]
    kind = want
    exclude = [(sc["cube"], 0.05), (sc["target"], 0.05)] + [(t["handle"], 0.06) for t in tools]
    obst = obstacle_points(meta, imgs, exclude)
    # remove the chosen tool's shaft points
    a = tool["tool_axis"]; h = tool["handle"][:2]
    rel = obst - h; s = rel @ a; l = rel @ np.array([-a[1], a[0]])
    obst = obst[~((s > -0.05) & (s < geo["L"] + 0.06) & (np.abs(l) < 0.10))]
    # also other tools stay as obstacles
    log("obstacle pts", len(obst))
    model = grasp_tool(rb, tool)
    for attempt in range(5):
        if rb.ended or time.time() > t_end: break
        meta, imgs, sc2 = scene_fn(rb)
        cube = sc2.get("cube", sc["cube"]); target = sc2.get("target", sc["target"])
        err = float(np.linalg.norm(cube[:2] - target[:2]))
        log("attempt", attempt, "cube", np.round(cube, 3), "target", np.round(target, 3), "err", round(err, 4))
        if err < 0.012:
            break
        gap = 0.03 if err > 0.04 else 0.02
        if kind == "hook":
            cands = [plan(model, geo, kind, cube, target, obst, 0.0, gap, lat) for lat in (0.04, 0.05)]
        else:
            cands = [plan(model, geo, kind, cube, target, obst, ang, gap, 0.0)
                     for ang in (0.0, 0.3, -0.3, 0.6, -0.6)]
        cands.sort(key=lambda c: c["cost"])
        pl = cands[min(attempt // 2, len(cands) - 1)] if attempt >= 2 else cands[0]
        log("plan", {k: (np.round(v, 3).tolist() if isinstance(v, np.ndarray) else v) for k, v in pl.items()})
        r = execute(rb, model, pl)
        log("exec", r)
        if rb.complete or rb.ended: break
    return True
