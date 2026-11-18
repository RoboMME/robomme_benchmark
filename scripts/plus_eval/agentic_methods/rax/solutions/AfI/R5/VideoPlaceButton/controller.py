"""VideoPlaceButton controller.

1. Read the conditioning video's robot-state trajectory, segment low-z dips into
   grasp / place / press events by gripper width, and take the first place after
   the button press as the goal xy.
2. Locate the red cube from RGB-D (front, then wrist close-up), align yaw, grasp,
   carry to the goal and release.
"""
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vision import parse_reply, backproject, color_mask, color_from_instruction, components, detect_targets  # noqa: E402
from geom import aa_to_R, grip_yaw, wrap_quarter, rot_error, min_area_yaw  # noqa: E402
from video import load_video_states, find_events, target_after_press, target_resting_at_press, target_last_before_press  # noqa: E402

GOAL_RULE = "first_after_press"  # v4: natural reading; ep1/ep2 failures attributed to high release (TCP z 0.024)

OUT = Path("/workspace/output")
SAFE_Z = 0.15
GRASP_Z = 0.016
PLACE_Z = 0.016   # expert releases at TCP z 0.015-0.021; target discs ~8mm tall visual
VMAX = 0.5          # max |action| per row for translation (0.025 m)
RMAX = 0.6          # max |action| per row for rotation (0.3 rad)


class Ended(Exception):
    pass


class Robot:
    def __init__(self, ctx, log):
        self.ctx, self.log = ctx, log
        self.meta = None
        self.ims = None
        self.ended = False
        self.complete = False
        self.grip_cmd = -1.0

    def observe(self):
        self.meta, self.ims = parse_reply(self.ctx.call("robomme_observe"))
        self._check(self.meta)
        return self.meta, self.ims

    @property
    def state(self):
        return np.array(self.meta["state"], float)

    def remaining(self):
        lim = self.meta.get("episode_limits") or {}
        return lim.get("remaining_episode_steps", 999)

    def _check(self, meta):
        tr = meta.get("transition") or {}
        for s in tr.get("steps", []) or []:
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        if meta.get("episode_ended"):
            self.ended = True
        if meta.get("task_complete") or tr.get("task_complete"):
            self.complete = True

    def step(self, rows):
        if self.ended:
            raise Ended()
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        rem = self.remaining()
        if rem is not None and rem <= 0:
            raise Ended()
        rows = rows[:max(1, min(100, rem))]
        reply = self.ctx.call("robomme_step", actions=rows)
        self.meta, self.ims = parse_reply(reply)
        self._check(self.meta)
        st = self.state
        self.log(f"step {self.meta.get('step_id')} n={len(rows)} pos={np.round(st[:3],3).tolist()} "
                 f"yaw={np.rad2deg(grip_yaw(st[3:6])):.1f} g={st[6]+st[7]:.3f} rem={self.remaining()} "
                 f"ended={self.ended} complete={self.complete}")
        if self.ended:
            raise Ended()
        return self.meta

    def move_to(self, xyz, yaw=None, tol=0.004, max_iter=12, vmax=VMAX, rows_per_call=None):
        """Closed-loop servo to xyz (and yaw) holding the current gripper command."""
        xyz = np.asarray(xyz, float)
        prev = None
        for it in range(max_iter):
            st = self.state
            err = xyz - st[:3]
            rerr = rot_error(st[3:6], yaw if yaw is not None else grip_yaw(st[3:6]))
            if prev is not None and np.linalg.norm(st[:3] - prev) < 0.0008 and np.linalg.norm(rerr) < 0.05:
                self.log("move_to: stalled")
                break
            prev = st[:3].copy()
            dist = np.linalg.norm(err)
            if dist < tol and np.linalg.norm(rerr) < 0.03:
                return True
            # number of rows needed so that each row is within limits
            n_t = int(np.ceil(np.max(np.abs(err)) / (0.05 * vmax))) if dist > 0 else 1
            n_r = int(np.ceil(np.max(np.abs(rerr)) / (0.5 * RMAX)))
            n = max(1, n_t, n_r)
            if rows_per_call:
                n = min(n, rows_per_call)
            # open-loop chunk: split error over n rows, but only send rows covering ~85%
            # then re-measure (controller targets measured pose + delta each row)
            d = err / n / 0.05
            r = rerr / n / 0.5
            row = [d[0], d[1], d[2], r[0], r[1], r[2], self.grip_cmd]
            self.step([row] * n)
        st = self.state
        return np.linalg.norm(xyz - st[:3]) < 3 * tol

    def gripper(self, cmd, n=8):
        self.grip_cmd = cmd
        self.step([[0, 0, 0, 0, 0, 0, cmd]] * n)


def locate_cube(robot, camera, color=None):
    color = color or getattr(robot, "color", "red")
    meta, ims = robot.meta, robot.ims
    rgb, dep = ims.get((camera, "rgb")), ims.get((camera, "depth"))
    if rgb is None or dep is None:
        return None
    comps = components(color_mask(rgb, color), min_area=8)
    if not comps:
        return None
    c, area = comps[0]
    vs, us = np.nonzero(c)
    p = backproject(meta, dep, camera, us, vs)
    p = p[np.all(np.isfinite(p), 1)]
    p = p[(p[:, 2] > -0.01) & (p[:, 2] < 0.12)]
    if len(p) < 6:
        return None
    ztop = np.percentile(p[:, 2], 95)
    top = p[p[:, 2] > ztop - 0.006]
    if len(top) < 6:
        top = p
    yaw, ctr, ext = min_area_yaw(top[:, :2])
    return dict(xy=ctr, ztop=float(ztop), yaw=float(yaw), ext=ext.tolist(), area=area, n_top=len(top))


def main(ctx):
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / "controller_log.txt", "a")
    t0 = time.time()

    def log(msg):
        line = f"[{time.time()-t0:7.1f}] {msg}"
        print(line)
        logf.write(line + "\n"); logf.flush()

    robot = Robot(ctx, log)
    try:
        run(robot, log)
    except Ended:
        log(f"episode ended complete={robot.complete}")
    except Exception as e:
        log("ERROR " + repr(e) + "\n" + traceback.format_exc())
    log(f"done ended={robot.ended} complete={robot.complete}")
    logf.close()


def choose_goal(robot, log):
    meta = robot.meta
    S, frames = load_video_states(meta)
    goal, why = None, "no_video"
    if S is not None and len(S) > 5:
        events = find_events(S)
        for e in events:
            log(f"event {e['kind']} xy={np.round(e['xy'],3).tolist()} idx={e['idx']} zmin={e['zmin']:.3f} "
                f"g {e['g_before']:.3f}->{e['g_after']:.3f}")
        instr = (meta.get("instruction") or "").lower()
        if "before" in instr and "after" not in instr:
            rule, alt = target_last_before_press, target_after_press
        else:
            rule, alt = target_after_press, target_last_before_press
        log(f"rule {rule.__name__}")
        goal, why = rule(events)
        if goal is None:
            # fallback: dips without gripper change count as presses
            ev2 = [dict(e, kind="press") if (e["kind"] == "other" and abs(e["g_before"] - e["g_after"]) < 0.01) else e
                   for e in events]
            goal, why = rule(ev2)
            why = "fallback_" + why
        if goal is None:
            goal, why = alt(ev2)
            why = "alt_" + why
    log(f"goal from video: {None if goal is None else np.round(goal,3).tolist()} ({why})")
    return goal


def run(robot, log):
    robot.observe()
    log("instruction: " + str(robot.meta.get("instruction")))
    robot.color = color_from_instruction(robot.meta.get("instruction"))
    log("cube colour: " + robot.color)
    log(f"state0 {np.round(robot.state,3).tolist()} limits {robot.meta.get('episode_limits')}")
    goal = choose_goal(robot, log)
    if goal is None:
        log("no goal found; abort")
        return

    cube = locate_cube(robot, "front")
    log(f"cube(front) {cube}")
    if cube is None:
        log("cube not found in front view; abort")
        return

    for attempt in range(3):
        ok = pick(robot, log, cube)
        if ok:
            break
        # re-detect cube after failed attempt
        robot.gripper(-1, 6)
        robot.move_to([robot.state[0], robot.state[1], SAFE_Z])
        robot.observe()
        c2 = locate_cube(robot, "front")
        log(f"re-detect cube(front) {c2}")
        if c2 is not None:
            cube = c2
    else:
        log("pick failed 3 times")
        return

    place(robot, log, goal)


def pick(robot, log, cube):
    st = robot.state
    robot.grip_cmd = -1.0
    # rise to safe height first if low
    if st[2] < SAFE_Z - 0.02:
        robot.move_to([st[0], st[1], SAFE_Z])
    robot.move_to([cube["xy"][0], cube["xy"][1], SAFE_Z], tol=0.006)
    # refine with wrist camera from above (camera is offset from TCP; it looks down)
    for k in range(2):
        robot.observe()
        cw = locate_cube(robot, "wrist")
        log(f"cube(wrist) {cw}")
        if cw is None:
            break
        if np.linalg.norm(cw["xy"] - cube["xy"]) < 0.06:
            cube = dict(cube, xy=cw["xy"], yaw=cw["yaw"], ztop=cw["ztop"])
        yaw_t = wrap_quarter(cube["yaw"])
        robot.move_to([cube["xy"][0], cube["xy"][1], SAFE_Z - 0.03], yaw=yaw_t, tol=0.004)
    yaw_t = wrap_quarter(cube["yaw"])
    gz = max(GRASP_Z, cube.get("ztop", 0.04) - 0.025)
    robot.move_to([cube["xy"][0], cube["xy"][1], 0.06], yaw=yaw_t, tol=0.004)
    robot.move_to([cube["xy"][0], cube["xy"][1], gz], yaw=yaw_t, tol=0.003, vmax=0.3)
    robot.gripper(1.0, 10)
    g = robot.state[6] + robot.state[7]
    log(f"grip width after close {g:.4f}")
    robot.move_to([robot.state[0], robot.state[1], SAFE_Z], yaw=yaw_t, vmax=0.4)
    g = robot.state[6] + robot.state[7]
    held = 0.015 < g < 0.06
    log(f"after lift width {g:.4f} held={held}")
    return held


def place(robot, log, goal):
    st = robot.state
    yaw = grip_yaw(st[3:6])
    robot.move_to([goal[0], goal[1], SAFE_Z], yaw=yaw, tol=0.004)
    robot.move_to([goal[0], goal[1], 0.07], yaw=yaw, tol=0.004)
    robot.move_to([goal[0], goal[1], PLACE_Z - 0.003], yaw=yaw, tol=0.003, vmax=0.3, max_iter=16)
    log(f"pre-release pos {np.round(robot.state[:3],4).tolist()} width {robot.state[6]+robot.state[7]:.4f}")
    robot.gripper(-1.0, 8)
    log(f"released at {np.round(robot.state[:3],3).tolist()}; complete={robot.complete}")
    robot.move_to([robot.state[0], robot.state[1], 0.12], yaw=yaw, tol=0.01)
    # hold still a few steps and let the evaluator register
    for _ in range(3):
        if robot.complete:
            break
        robot.step([[0, 0, 0, 0, 0, 0, -1]] * 5)
