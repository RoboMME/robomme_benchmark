"""MoveCube controller: classify manner from conditioning-video robot states, then
execute hook-pull (tool), pick-and-place, or gripper-push with closed-loop checks."""
import sys, os, json, math, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from motion import Robot, move_to, hold, wrap, log, LOG
from scene import locate
from perc import world_points

OUT = "/workspace/output"
Z_SAFE = 0.14


def classify_video(meta0):
    """Use robot states in the conditioning video (phase video_demo)."""
    feats = {"manner": "hook"}
    try:
        seq = meta0.get("frame_sequence") or {}
        man = json.load(open(os.path.join("/workspace", seq["manifest_path"])))
        st = [np.array(f["state"]) for f in man["frames"] if f.get("phase") == "video_demo"]
        if len(st) < 5:
            feats["manner"] = "unknown"; return feats
        st = np.array(st)
        w = st[:, 6] + st[:, 7]
        closed = np.nonzero(w < 0.075)[0]
        feats["min_w"] = float(w.min())
        if len(closed) == 0:
            feats["manner"] = "push_open"; return feats
        i0 = closed[0]
        held = float(np.median(w[closed]))
        feats["held_w"] = held
        feats["grasp_pos"] = st[i0, :3].tolist()
        zc = st[closed, 2]
        feats["z_closed_max"] = float(zc.max())
        xy = st[closed, :2]
        feats["xy_travel_closed"] = float(np.linalg.norm(xy[-1] - xy[0]))
        # low-z travel while closed
        low = closed[st[closed, 2] < 0.04]
        feats["low_travel"] = float(np.linalg.norm(st[low[-1], :2] - st[low[0], :2])) if len(low) > 1 else 0.0
        if held < 0.006:
            feats["manner"] = "push_closed"
        elif held > 0.034:
            feats["manner"] = "pick"
        elif held < 0.0232:
            feats["manner"] = "peg"
        elif held < 0.0262:
            feats["manner"] = "scoop"
        else:
            feats["manner"] = "hook"
    except Exception as e:
        feats["error"] = repr(e)
    return feats


def save_json(name, obj):
    try:
        os.makedirs(OUT, exist_ok=True)
        json.dump(obj, open(os.path.join(OUT, name), "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), indent=1)
    except Exception:
        pass


def scene(rb):
    meta, imgs = rb.observe()
    sc = locate(meta, imgs)
    log("scene", {k: (np.round(v, 3).tolist() if isinstance(v, np.ndarray) else v) for k, v in sc.items()})
    return meta, imgs, sc


def bar_geometry(meta, imgs, sc):
    """Bar (hook) position along tool axis and side, from the front image."""
    geo = {"s_tip": sc.get("tool_len", 0.29), "side": 1.0, "blen": 0.06}
    try:
        rgb = imgs[("front", "rgb")].astype(int)
        P, valid = world_points(meta, "front", imgs[("front", "depth")])
        a = sc["tool_axis"]; n = np.array([-a[1], a[0]]); h = sc["handle"][:2]
        rel = P[..., :2] - h; s = rel @ a; l = rel @ n
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        pink = (r > 130) & (r - g > 35) & (r - g < 90) & (g - b < 32) & (P[..., 2] > 0.005) & (P[..., 2] < 0.04)
        m = pink & (s > sc["tool_len"] - 0.1) & (s < sc["tool_len"] + 0.05) & (np.abs(l) < 0.12)
        if m.sum() >= 8:
            lm = l[m]
            side = 1.0 if np.median(lm) > 0 else -1.0
            geo["side"] = side
            geo["blen"] = float(np.clip(np.percentile(np.abs(lm), 95), 0.04, 0.09))
            geo["s_tip"] = float(np.percentile(s[m], 50))
        geo["pink_px"] = int(m.sum())
    except Exception as e:
        geo["error"] = repr(e)
    return geo


def grasp_tool(rb, sc):
    a = sc["tool_axis"]; alpha = math.atan2(a[1], a[0])
    gy = wrap(alpha)
    if gy > math.pi / 2: gy -= math.pi
    if gy < -math.pi / 2: gy += math.pi
    h = sc["handle"]; z = max(0.012, sc["handle_top"] - 0.008)
    move_to(rb, [rb.pos()[0], rb.pos()[1], Z_SAFE], None, -1, max_steps=40)
    move_to(rb, [h[0], h[1], Z_SAFE], gy, -1, max_steps=80)
    move_to(rb, [h[0], h[1], z + 0.03], gy, -1, max_steps=40)
    move_to(rb, [h[0], h[1], z], gy, -1, max_steps=40, vmax=0.4)
    hold(rb, 8, 1.0)
    w = rb.grip_open()
    tcp = rb.pos()
    model = {"off": wrap(alpha - rb.yaw()), "z_grasp": float(tcp[2]),
             "tcp_s": float((tcp[:2] - h[:2]) @ a), "tcp_l": float((tcp[:2] - h[:2]) @ np.array([-a[1], a[0]])),
             "w": w}
    log("grasp", model)
    move_to(rb, [tcp[0], tcp[1], Z_SAFE], None, 1.0, max_steps=50)
    model["held"] = rb.grip_open() > 0.01
    return model


def tool_point(tcp_xy, yaw, model, s, l):
    """World xy of tool-frame point (s along axis from handle centre, l lateral)."""
    ang = yaw + model["off"]
    a = np.array([math.cos(ang), math.sin(ang)]); n = np.array([-a[1], a[0]])
    hc = tcp_xy - a * model["tcp_s"] - n * model["tcp_l"]
    return hc + a * s + n * l


def hook_pull(rb, model, geo, cube, target, lat=0.035, gap=0.03):
    D = cube[:2] - target[:2]; dist = float(np.linalg.norm(D))
    d = D / max(dist, 1e-6)
    yaw = wrap(math.atan2(d[1], d[0]) - model["off"])
    n = np.array([-d[1], d[0]]) * geo["side"]
    s_c = geo["s_tip"] - 0.01 - 0.02 - gap
    # handle centre so that cube sits at (s_c, lat) in tool frame (bar side)
    hc = cube[:2] - d * s_c - n * lat
    a = d; nn = np.array([-a[1], a[0]])
    tcp = hc + a * model["tcp_s"] + nn * model["tcp_l"]
    log("hook plan yaw", yaw, "tcp", tcp, "dist", dist)
    move_to(rb, [rb.pos()[0], rb.pos()[1], Z_SAFE], None, 1.0, max_steps=40)
    move_to(rb, [rb.pos()[0], rb.pos()[1], Z_SAFE], yaw, 1.0, max_steps=60)
    move_to(rb, [tcp[0], tcp[1], Z_SAFE], yaw, 1.0, max_steps=80)
    zt = model["z_grasp"] + 0.004
    move_to(rb, [tcp[0], tcp[1], zt + 0.04], yaw, 1.0, max_steps=40)
    move_to(rb, [tcp[0], tcp[1], zt], yaw, 1.0, max_steps=40, vmax=0.3)
    zr = rb.pos()[2]
    log("lowered z", zr, "target", zt)
    if zr > zt + 0.012:
        return "blocked"
    p0 = rb.pos(); goal = p0.copy(); goal[:2] -= d * (dist + gap); goal[2] = zt
    nseg = max(2, int(math.ceil((dist + gap) / 0.03)))
    for k in range(1, nseg + 1):
        move_to(rb, p0 + (goal - p0) * k / nseg, yaw, 1.0, max_steps=25, vmax=0.3)
    hold(rb, 3, 1.0)
    p = rb.pos()
    move_to(rb, [p[0], p[1], p[2] + 0.05], yaw, 1.0, max_steps=20)
    move_to(rb, [p[0] - 0.0, p[1], Z_SAFE], yaw, 1.0, max_steps=30)
    return "pulled"


def run_hook(rb, t_end):
    meta, imgs, sc = scene(rb)
    if "handle" not in sc or "cube" not in sc or "target" not in sc:
        log("hook: missing", list(sc)); return False
    geo = bar_geometry(meta, imgs, sc); log("bar", geo)
    model = grasp_tool(rb, sc)
    for attempt in range(4):
        if rb.ended or time.time() > t_end: break
        meta, imgs, sc2 = scene(rb)
        cube = sc2.get("cube", sc["cube"]); target = sc2.get("target", sc["target"])
        err = float(np.linalg.norm(cube[:2] - target[:2]))
        log("attempt", attempt, "cube-target err", err)
        if err < 0.015 and attempt > 0:
            break
        lat = 0.035 + 0.012 * attempt * (1 if attempt % 2 else 1)
        r = hook_pull(rb, model, geo, cube, target, lat=lat)
        log("hook result", r)
        if rb.complete: break
    return True


def pick_place(rb):
    for attempt in range(4):
        if rb.ended: return
        meta, imgs, sc = scene(rb)
        if "cube" not in sc or "target" not in sc:
            log("pick: missing", list(sc)); return
        c = sc["cube"]; t = sc["target"]
        if np.linalg.norm(c[:2] - t[:2]) < 0.012 and attempt > 0:
            return
        yaw = [0.0, math.pi / 4, -math.pi / 4, math.pi / 8][attempt]
        zc = max(0.012, c[2] - 0.022)
        move_to(rb, [rb.pos()[0], rb.pos()[1], Z_SAFE], None, -1, max_steps=40)
        move_to(rb, [c[0], c[1], Z_SAFE], yaw, -1, max_steps=80)
        move_to(rb, [c[0], c[1], zc + 0.03], yaw, -1, max_steps=40)
        move_to(rb, [c[0], c[1], zc], yaw, -1, max_steps=40, vmax=0.4)
        hold(rb, 8, 1.0)
        move_to(rb, [c[0], c[1], 0.12], None, 1.0, max_steps=40)
        w = rb.grip_open(); log("pick attempt", attempt, "yaw", yaw, "grip", w)
        if w < 0.01:
            hold(rb, 5, -1.0)
            continue
        move_to(rb, [t[0], t[1], 0.12], None, 1.0, max_steps=100)
        move_to(rb, [t[0], t[1], zc + 0.008], None, 1.0, max_steps=50, vmax=0.4)
        hold(rb, 8, -1.0)
        move_to(rb, [t[0], t[1], 0.15], None, -1.0, max_steps=40)


def push(rb, grip):
    for attempt in range(3):
        meta, imgs, sc = scene(rb)
        c = sc["cube"]; t = sc["target"]
        D = t[:2] - c[:2]; dist = float(np.linalg.norm(D))
        if dist < 0.015: break
        d = D / dist
        yaw = wrap(math.atan2(d[1], d[0]) + math.pi / 2)
        if yaw > math.pi / 2: yaw -= math.pi
        if yaw < -math.pi / 2: yaw += math.pi
        start = c[:2] - d * 0.06
        move_to(rb, [rb.pos()[0], rb.pos()[1], Z_SAFE], yaw, grip, max_steps=40)
        move_to(rb, [start[0], start[1], Z_SAFE], yaw, grip, max_steps=80)
        move_to(rb, [start[0], start[1], 0.02], yaw, grip, max_steps=60, vmax=0.5)
        goal = t[:2] - d * 0.03
        p0 = rb.pos(); g = np.array([goal[0], goal[1], 0.02])
        nseg = max(2, int(math.ceil(np.linalg.norm(g[:2] - p0[:2]) / 0.03)))
        for k in range(1, nseg + 1):
            move_to(rb, p0 + (g - p0) * k / nseg, yaw, grip, max_steps=25, vmax=0.3)
        p = rb.pos()
        move_to(rb, [p[0] - d[0] * 0.04, p[1] - d[1] * 0.04, 0.02], yaw, grip, max_steps=20)
        move_to(rb, [rb.pos()[0], rb.pos()[1], Z_SAFE], yaw, grip, max_steps=40)


def main(ctx):
    t0 = time.time(); t_end = t0 + 1000
    rb = Robot(ctx)
    try:
        reply = ctx.call("robomme_observe")
        meta0 = json.loads(reply["content"][0]["text"])
        feats = classify_video(meta0)
        log("video feats", feats)
        save_json("video_feats.json", feats)
        m = feats["manner"]
        if m in ("hook", "peg", "scoop", "unknown"):
            from tooluse import run_tool
            run_tool(rb, m if m != "unknown" else "hook", t_end, scene)
        elif m == "pick":
            pick_place(rb)
        elif m == "push_closed":
            push(rb, 1.0)
        else:
            push(rb, -1.0)
        if not rb.ended:
            meta, imgs, sc = scene(rb)
    except Exception:
        log("EXC", traceback.format_exc())
    finally:
        log("end step", rb.step_id, "complete", rb.complete, "ended", rb.ended, "t", time.time() - t0)
        try:
            os.makedirs(OUT, exist_ok=True)
            open(os.path.join(OUT, "log.txt"), "w").write("\n".join(LOG))
        except Exception:
            pass
