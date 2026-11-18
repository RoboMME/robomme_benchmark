"""Task skills: locate and pick containers."""
import numpy as np
import perception as P
import motion as M


def observe(robot):
    reply = robot.ctx.call("robomme_observe")
    obs = P.decode_reply(reply)
    robot._update(obs["meta"])
    return obs


def cam_offset(obs, cam="wrist"):
    m = obs["meta"]
    return np.array(m["camera_poses"][cam]["position_m"]) - np.array(m["state"][:3])


def nearest(cs, xy, max_d=0.06):
    if not cs:
        return None, 1e9
    d = [np.linalg.norm(c["xy"] - np.asarray(xy)[:2]) for c in cs]
    i = int(np.argmin(d))
    return (cs[i], d[i]) if d[i] < max_d else (None, d[i])


def look_at(robot, xy, z=0.25, log=print):
    """Place the wrist camera above xy; return wrist observation."""
    obs = observe(robot)
    off = cam_offset(obs)
    tgt = np.array([xy[0] - off[0], xy[1] - off[1], z])
    robot.move_to(tgt, speed=1.0, tol=0.006)
    robot.hold(3)
    return observe(robot)


def refine_target(robot, xy, log=print):
    obs = look_at(robot, xy)
    cs = [c for c in P.find_containers(obs, cams=("wrist",), exclude_xy=None)
          if 0.05 <= c["ztop"] <= 0.095 and c["extent"].max() <= 0.10 and len(c["top"]) >= 8]
    c, d = nearest(cs, xy, max_d=0.045)
    if c is None:
        log(f"refine: no wrist container near {np.round(xy,3)} (nearest {d:.3f})")
        return None, obs
    yaw, area = P.estimate_yaw(c["top"][:, :2]) if len(c["top"]) > 20 else (None, 0)
    log(f"refine: xy {np.round(c['xy'],3)} ztop {c['ztop']:.3f} n {c['n']} top {len(c['top'])} "
        f"ext {np.round(c['extent'],3)} yaw {None if yaw is None else round(np.degrees(yaw),1)}")
    return {"xy": c["xy"], "ztop": c["ztop"], "yaw": yaw}, obs


def pick(robot, tgt, lift_z=0.22, grasp_depth=0.025, log=print):
    """Grasp the container at tgt (xy, ztop, yaw) and lift it. Returns True if held."""
    cur = M.gripper_yaw(robot.state)
    if tgt.get("yaw") is not None:
        # square symmetric: choose equivalent yaw closest to current
        # square symmetric: equivalent yaw closest to the home yaw (0) keeps the wrist in range
        y = M.wrap(tgt["yaw"], np.pi / 2)
        robot.rotate_yaw(y)
    robot.gripper = -1.0
    x, yv = tgt["xy"]
    ztop = tgt["ztop"]
    robot.move_to([x, yv, ztop + 0.08], speed=0.8, tol=0.004)
    robot.move_to([x, yv, ztop + 0.03], speed=0.4, tol=0.003)
    robot.move_to([x, yv, max(0.02, ztop - grasp_depth)], speed=0.25, tol=0.003)
    robot.set_gripper(1.0, 15)
    width = float(robot.state[6] + robot.state[7])
    log(f"pick: closed width {width:.4f} at {np.round(robot.pos,3)}")
    if width < 0.008:
        log("pick: gripper closed on nothing")
        return False
    robot.move_to([x, yv, lift_z], speed=0.5, tol=0.006)
    robot.hold(3)
    width = float(robot.state[6] + robot.state[7])
    held = width > 0.01
    log(f"pick: lifted to {np.round(robot.pos,3)} width {width:.4f} held={held}")
    return held


def place(robot, xy, ztop, release_above=0.005, log=print):
    """Put a held container down with its top at ~ztop over xy and release."""
    x, y = xy
    robot.move_to([x, y, ztop + 0.05], speed=0.6, tol=0.005)
    robot.move_to([x, y, ztop - 0.025 + release_above], speed=0.25, tol=0.003)
    robot.set_gripper(-1.0, 10)
    robot.move_to([x, y, ztop + 0.08], speed=0.6, tol=0.006)
    log(f"place: released at {np.round(xy,3)}")
