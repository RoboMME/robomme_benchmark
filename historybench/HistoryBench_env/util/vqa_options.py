from typing import Callable, Dict, List
import numpy as np

from historybench.HistoryBench_env.util.planner import (
    grasp_and_lift_peg_side,
    insert_peg,
    solve_button,
    solve_button_ready,
    solve_hold_obj,
    solve_hold_obj_absTimestep,
    solve_pickup,
    solve_pickup_bin,
    solve_push_to_target,
    solve_push_to_target_with_peg,
    solve_putdown_whenhold,
    solve_putonto_whenhold,
    solve_putonto_whenhold_binspecial,
    solve_swingonto,
    solve_swingonto_withDirection,
    solve_swingonto_whenhold,
    solve_strong_reset,
)


def _options_default(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = []

    return options


def _options_videorepick(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = [
        {
            "label": "pick up the cube (click segmentation)",
            "solve": lambda require_target=require_target: solve_pickup(
                env, planner, obj=require_target()
            ),
        },
        {
            "label": "put it down",
            "solve": lambda require_target=require_target: solve_putdown_whenhold(
                env, planner, obj=require_target(), release_z=0.01
            ),
        },
    ]
    button_obj = getattr(base, "button_left", None)
    if button_obj is not None:
        options.append(
            {
                "label": "press the button to finish",
                "solve": lambda button_obj=button_obj: solve_button(
                    env, planner, obj=button_obj
                ),
            }
        )
    return options


def _options_binfill(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = [
        {
            "label": "pickup (click segmentation)",
            "solve": lambda require_target=require_target: solve_pickup(
                env, planner, obj=require_target()
            ),
        },
    ]
    target = getattr(base, "board_with_hole", None)
    if target is not None:
        options.append(
            {
                "label": "put it into the bin",
                "solve": lambda require_target=require_target, target=target: solve_putonto_whenhold_binspecial(
                    env, planner, obj=require_target(), target=target
                ),
            }
        )
    button_obj = getattr(base, "button", None)
    if button_obj is not None:
        options.append(
            {
                "label": "press the button",
                "solve": lambda button_obj=button_obj: solve_button(
                    env, planner, obj=button_obj
                ),
            }
        )
    return options


def _options_button_unmask(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = []
    button_obj = getattr(base, "button_left", None) or getattr(base, "button", None)
    if button_obj is not None:
        options.append(
            {
                "label": "press the button",
                "solve": lambda button_obj=button_obj: solve_button(
                    env, planner, obj=button_obj
                ),
            }
        )

    options.extend([{
            "label": "pick up the container (click segmentation)",
            "solve": lambda require_target=require_target: solve_pickup_bin(
                env, planner, obj=require_target()
            ),
        },
        {
            "label": "put down the container",
            "solve": lambda require_target=require_target: solve_putdown_whenhold(
                env, planner, obj=require_target(), release_z=0.01
            ),
        },])
    return options

def _options_button_unmask_swap(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = []
    button_obj_left = getattr(base, "button_left", None) or getattr(base, "button", None)
    if button_obj_left is not None:
        options.append(
            {
                "label": "press the left button",
                "solve": lambda button_obj=button_obj_left: solve_button(
                    env, planner, obj=button_obj
                ),
            }
        )
    button_obj_right = getattr(base, "button_right", None)
    if button_obj_right is not None:
        options.append(
            {
                "label": "press the right button",
                "solve": lambda button_obj=button_obj_right: solve_button(
                    env, planner, obj=button_obj
                ),
            }
        )


    options.extend([{
            "label": "pick up the container(click segmentation)",
            "solve": lambda require_target=require_target: solve_pickup_bin(
                env, planner, obj=require_target()
            ),
        },
        {
            "label": "put down the container",
            "solve": lambda require_target=require_target: solve_putdown_whenhold(
                env, planner, obj=require_target(), release_z=0.01
            ),
        },])
    return options 


def _options_insertpeg(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = []

    options.append(
        {
            "label": "pick up the peg by grasping one end (click segmentation)",
            "solve": lambda require_target=require_target: grasp_and_lift_peg_side(
                env, planner, obj=require_target()
            ),
        }
    )

    options.append(
        {
            "label": "Insert the peg from the right side",
            "solve": lambda direction=1: insert_peg(
                env,
                planner,
                direction=direction,
                obj=env.obj_flag,
                insert_obj=env.insert_target,
                cut_retreat=True,
            ),
        }
    )
    options.append(
        {
            "label": "Insert the peg from the left side",#和subgoal保持一致
            "solve": lambda direction=-1: insert_peg(
                env,
                planner,
                direction=direction,
                obj=env.obj_flag,
                insert_obj=env.insert_target,
                cut_retreat=True,
            ),
        }
    )

    return options


def _options_movecube(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = []
    cube = getattr(base, "cube", None)
    cube_goal = getattr(base, "goal_site", None)
    peg_target = getattr(base, "grasp_target", None)
    obj_flag = getattr(base, "obj_flag", None)
    direction1 = getattr(base, "direction1", None)
    direction2 = getattr(base, "direction2", None)

    if peg_target is not None:
        options.append(
            {
                "label": "Pick up the peg",
                "solve": lambda peg_target=peg_target: grasp_and_lift_peg_side(
                    env, planner, obj=peg_target
                ),
            }
        )



    if cube is not None and cube_goal is not None and direction2 is not None and obj_flag is not None:
        options.append(
            {
                "label": "Hook the cube to the target with the peg",
                "solve": lambda cube=cube, goal=cube_goal, direction=direction2, obj_flag=obj_flag: solve_push_to_target_with_peg(
                    env,
                    planner,
                    obj=cube,
                    target=goal,
                    direction=direction,
                    obj_flag=obj_flag,
                ),
            }
        )


    if cube is not None and cube_goal is not None:
        options.append(
            {
                "label": "Close gripper and push the cube to the target",
                "solve": lambda cube=cube, goal=cube_goal: solve_push_to_target(
                    env, planner, obj=cube, target=goal
                ),
            }
        )
    if cube is not None and cube_goal is not None:
        options.append(
            {
                "label": "Pick up the cube",
                "solve": lambda cube=cube: solve_pickup(env, planner, obj=cube),
            }
        )
        options.append(
            {
                "label": "Place the cube onto the target",
                "solve": lambda cube=cube, goal=cube_goal: solve_putonto_whenhold(
                    env, planner, obj=cube, target=goal
                ),
            }
        )

    return options


def _options_patternlock(env, planner, require_target, base) -> List[dict]:
    """
    Dynamically pick the nearest target to the current TCP and move to the
    closest neighbour along the requested direction.
    """
    directions = [
        "forward",
        "backward",
        "left",
        "right",
        "forward-left",
        "forward-right",
        "backward-left",
        "backward-right",
    ]

    dir_vectors = {
        "forward": np.array([1.0, 0.0]),
        "backward": np.array([-1.0, 0.0]),
        "left": np.array([0.0, 1.0]),
        "right": np.array([0.0, -1.0]),
        "forward-left": np.array([1.0, 1.0]) / np.sqrt(2.0),
        "forward-right": np.array([1.0, -1.0]) / np.sqrt(2.0),
        "backward-left": np.array([-1.0, 1.0]) / np.sqrt(2.0),
        "backward-right": np.array([-1.0, -1.0]) / np.sqrt(2.0),
    }

    # Quick half-plane filters so we only consider targets in the intended direction.
    eps = 1e-6
    dir_filters = {
        "forward": lambda dx, dy: dx > eps,
        "backward": lambda dx, dy: dx < -eps,
        "left": lambda dx, dy: dy > eps,
        "right": lambda dx, dy: dy < -eps,
        "forward-left": lambda dx, dy: dx > eps and dy > eps,
        "forward-right": lambda dx, dy: dx > eps and dy < -eps,
        "backward-left": lambda dx, dy: dx < -eps and dy > eps,
        "backward-right": lambda dx, dy: dx < -eps and dy < -eps,
    }

    def _actor_xy(actor):
        pose = getattr(actor, "pose", None)
        if pose is None:
            pose = actor.get_pose() if hasattr(actor, "get_pose") else None
        pos = getattr(pose, "p", None)
        if pos is None:
            return None
        arr = np.asarray(pos).reshape(-1)
        return arr[:2] if arr.size >= 2 else None

    def _collect_targets():
        """
        Gather all available target actors (targets_grid/buttons_grid/selected_buttons)
        and deduplicate by object id.
        """
        buckets = (
            getattr(base, "targets_grid", None),
            getattr(base, "buttons_grid", None),
            getattr(base, "selected_buttons", None),
        )
        seen = set()
        targets = []
        for bucket in buckets:
            if not bucket:
                continue
            for t in bucket:
                if t is None:
                    continue
                t_id = id(t)
                if t_id in seen:
                    continue
                seen.add(t_id)
                targets.append(t)
        return targets

    def _closest_target_to_tcp():
        targets = _collect_targets()
        if not targets:
            raise ValueError("PatternLock requires targets_grid/buttons_grid to be initialized.")

        tcp_pose = np.asarray(env.agent.tcp.pose.p).reshape(-1)
        if tcp_pose.size < 2:
            raise ValueError("TCP pose does not provide x/y coordinates.")
        tcp_xy = tcp_pose[:2]

        dist_list = []
        for t in targets:
            t_xy = _actor_xy(t)
            if t_xy is None:
                continue
            dist = float(np.linalg.norm(t_xy - tcp_xy))
            dist_list.append((dist, t, t_xy))

        if not dist_list:
            raise ValueError("PatternLock could not compute any valid target positions.")

        dist_list.sort(key=lambda item: item[0])
        return dist_list[0][1], dist_list[0][2], targets

    def _target_for_direction(dir_label: str):
        ref_target, ref_xy, candidates = _closest_target_to_tcp()
        vec = dir_vectors[dir_label]
        filt = dir_filters[dir_label]

        best = None
        best_score = None
        for cand in candidates:
            if cand is ref_target:
                continue
            c_xy = _actor_xy(cand)
            if c_xy is None:
                continue
            delta = c_xy - ref_xy
            if delta.shape[0] < 2:
                continue
            dx, dy = float(delta[0]), float(delta[1])
            if not filt(dx, dy):
                continue
            dist = float(np.linalg.norm(delta))
            if dist < eps:
                continue
            align = float(np.dot(delta / dist, vec))
            score = (-align, dist)  # prioritize alignment, then closeness
            if best_score is None or score < best_score:
                best_score = score
                best = cand

        if best is None:
            print(f"[PatternLock] No candidate in direction '{dir_label}', using nearest target.")
            best = ref_target
        return best

    def _solve_direction(chosen_dir: str):
        try:
            target = _target_for_direction(chosen_dir)
        except ValueError as e:
            print(f"[PatternLock] {e}")
            return

        record_flag = getattr(base, "swing_qpos", None) is None
        return solve_swingonto(env, planner, target=target, record_swing_qpos=record_flag)

    options: List[dict] = []
    for dir_label in directions:
        options.append(
            {
                "label": f"move {dir_label}",
                "solve": (lambda d=dir_label: _solve_direction(d)),
            }
        )
    return options


def _options_routestick(env, planner, require_target, base) -> List[dict]:
    def _solve_route(side: str, direction: str):
        """
        Choose a target stick based on the desired side and move around it
        using the requested rotation direction.
        - side: "left"/"right" → pick the nearest target on that side of the
          last visited stick (fallback: current gripper position).
        - direction: "clockwise"/"counterclockwise" (accept short aliases).
        """
        # Normalize direction text
        dir_map = {
            "clock": "clockwise",
            "clockwise": "clockwise",
            "cw": "clockwise",
            "counterclock": "counterclockwise",
            "counterclockwise": "counterclockwise",
            "anticlockwise": "counterclockwise",
            "ccw": "counterclockwise",
        }
        direction_norm = dir_map.get(str(direction).lower(), "counterclockwise")

        def _actor_xy(a):
            if a is None:
                return None
            pose = getattr(a, "pose", None)
            if pose is None:
                pose = a.get_pose()
            p = getattr(pose, "p", None)
            if p is None:
                return None
            arr = np.asarray(p).reshape(-1)
            return arr[:2] if arr.size >= 2 else None

        # Reference actor: allowed button nearest to current TCP.
        ref_actor = None

        # Candidate targets: strictly use allowed indices on the buttons grid.
        grid = list(getattr(base, "buttons_grid", []) or [])
        allowed_idx = [int(i) for i in getattr(base, "route_button_indices", [0, 2, 4, 6, 8])]
        allowed_set = set(allowed_idx)
        allowed_candidates = [
            (idx, actor) for idx, actor in enumerate(grid) if idx in allowed_set and actor is not None
        ]

        # Pick reference as the allowed button closest to the current TCP.
        tcp_pose = np.asarray(env.agent.tcp.pose.p).reshape(-1)
        tcp_xy = tcp_pose[:2] if tcp_pose.size >= 2 else None
        dist_list = []
        for idx, actor in allowed_candidates:
            xy = _actor_xy(actor)
            if xy is None or tcp_xy is None:
                continue
            dist_list.append((np.linalg.norm(xy - tcp_xy), idx, actor))

        if dist_list:
            _, ref_idx, ref_actor = min(dist_list, key=lambda item: item[0])
        elif allowed_candidates:
            ref_idx, ref_actor = allowed_candidates[0]
        else:
            ref_idx, ref_actor = 0, None

        candidates = [(idx, actor) for idx, actor in allowed_candidates if actor is not ref_actor]

        if not candidates:
            raise ValueError("RouteStick: no available targets to swing onto.")

        side_l = str(side).lower()
        side_candidates = []
        for idx, actor in candidates:
            if side_l == "left" and idx > ref_idx:
                side_candidates.append((idx, actor))
            elif side_l == "right" and idx < ref_idx:
                side_candidates.append((idx, actor))

        
        if not side_candidates:
            raise ValueError("RouteStick: failed to determine a target for the selected side.")

        # Pick the closest candidate by index in the requested direction (no distance check).
        if side_l == "left":
            _target_idx, target = min(side_candidates, key=lambda item: item[0] - ref_idx)
        else:
            _target_idx, target = min(side_candidates, key=lambda item: ref_idx - item[0])

        swing_radius = float(getattr(base, "swing_radius", 0.2) or 0.2)
        return solve_swingonto_withDirection(
            env, planner, target=target, radius=swing_radius, direction=direction_norm
        )

    options: List[dict] = []
    option_defs = [
        ("move to the nearest left target by circling around the stick clockwise", "left", "clockwise"),
        ("move to the nearest right target by circling around the stick clockwise", "right", "clockwise"),
        ("move to the nearest left target by circling around the stick counterclockwise", "left", "counterclockwise"),
        ("move to the nearest right target by circling around the stick counterclockwise", "right", "counterclockwise"),
    ]
    for label, side, direction in option_defs:
        options.append(
            {
                "label": label,
                "solve": (lambda s=side, d=direction: _solve_route(s, d)),
            }
        )
    return options

def _options_pickhighlight(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = []

    button_obj = getattr(base, "button", None) or getattr(base, "button_left", None)
    if button_obj is not None:
        options.append(
            {
                "label": "press button",
                "solve": lambda button_obj=button_obj: solve_button(
                    env, planner, obj=button_obj
                ),
            }
        )

    target_cubes = list(getattr(base, "target_cubes", []) or [])
    target_labels = list(getattr(base, "target_labels", []) or []) or getattr(
        base, "target_cube_names", []
    )

    options.extend(
        [
            {
                "label": "pick up the highlighted cube (click segmentation)",
                "solve": lambda require_target=require_target: solve_pickup(
                    env, planner, obj=require_target()
                ),
            },
            {
                "label": "place the cube onto the table",
                "solve": lambda require_target=require_target: solve_putdown_whenhold(
                    env, planner, obj=require_target(), release_z=0.01
                ),
            },
        ]
    )

    return options

def _options_pickxtimes(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = []
    options.extend(
        [
            {
                "label": "pick up the cube (click segmentation)",
                "solve": lambda require_target=require_target: solve_pickup(
                    env, planner, obj=require_target()
                ),
            },
            {
                "label": "place the cube onto the target (click segmentation)",
                "solve": lambda require_target=require_target: solve_putonto_whenhold(
                    env, planner, obj=require_target(), target=env.target
                ),
            },
            {
                "label": "press the button to stop",
                "solve": lambda: solve_button(
                    env, planner, obj=env.button
                ),
            }
        ]
    )

    return options

def _options_swingxtimes(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = []

    options.append(
        {
            "label": "pick up the cube (click segmentation)",
            "solve": lambda require_target=require_target: solve_pickup(
                env, planner, obj=require_target()
            ),
        }
    )

    target_cube = getattr(base, "target_cube", None)
    if target_cube is not None:
        options.append(
            {
                "label": "move to the top of the target (click segmentation)",
                "solve": lambda require_target=require_target, target_cube=target_cube: solve_swingonto_whenhold(
                    env,
                    planner,
                    obj=target_cube,
                    target=require_target(),
                    height=0.1,
                ),
            }
        )
        options.append(
            {
                "label": "put the cube on the table",
                "solve": lambda target_cube=target_cube: solve_putdown_whenhold(
                    env, planner, obj=target_cube
                ),
            }
        )

    button_obj = getattr(base, "button", None) or getattr(base, "button_left", None)
    if button_obj is not None:
        options.append(
            {
                "label": "press the button",
                "solve": lambda button_obj=button_obj: solve_button(
                    env, planner, obj=button_obj
                ),
            }
        )

    return options

def _options_videoplaceorder(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = [
        {
            "label": "pick up the cube (click segmentation)",
            "solve": lambda require_target=require_target: solve_pickup(
                env, planner, obj=require_target()
            ),
        },
    ]

    target_cube = getattr(base, "target_cube", None)
    if target_cube is not None:
        options.append(
            {
                "label": "droponto (click segmentation to choose target)",
                "solve": lambda require_target=require_target, target_cube=target_cube: solve_putonto_whenhold(
                    env, planner, obj=target_cube, target=require_target()
                ),
            }
        )


    return options
def _options_videoplacebutton(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = [
        {
            "label": "pick up the cube (click segmentation)",
            "solve": lambda require_target=require_target: solve_pickup(
                env, planner, obj=require_target()
            ),
        },
    ]

    target_cube = getattr(base, "target_cube", None)
    if target_cube is not None:
        options.append(
            {
                "label": "droponto (click segmentation to choose target)",
                "solve": lambda require_target=require_target, target_cube=target_cube: solve_putonto_whenhold(
                    env, planner, obj=target_cube, target=require_target()
                ),
            }
        )

    return options


def _options_stopcube(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = []
    button_obj = getattr(base, "button", None)

    if button_obj is not None:
        options.append(
            {
                "label": "move to the top of the button to prepare",
                "solve": lambda button_obj=button_obj: solve_button_ready(
                    env, planner, obj=button_obj
                ),
            }
        )

    steps_press = getattr(base, "steps_press", None)
    if steps_press is not None:
        interval = getattr(base, "interval", 30)
        abs_timestep = max(0, int(steps_press - interval))
        options.append(
            {
                "label": "remain static",
                "solve": lambda abs_timestep=abs_timestep: solve_hold_obj_absTimestep(
                    env, planner, absTimestep=abs_timestep
                ),
            }
        )

    if button_obj is not None:
        options.append(
            {
                "label": "press button to stop the cube",
                "solve": lambda button_obj=button_obj: solve_button(
                    env, planner, obj=button_obj, without_hold=True
                ),
            }
        )

    return options
def _options_video_unmask(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = []
    options.extend([{
            "label": "pick up the container (click segmentation)",
            "solve": lambda require_target=require_target: solve_pickup_bin(
                env, planner, obj=require_target()
            ),
        },
        {
            "label": "put down the container",
            "solve": lambda require_target=require_target: solve_putdown_whenhold(
                env, planner, obj=require_target(), release_z=0.01
            ),
        },])
    return options

def _options_video_unmask_swap(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = []
    options: List[dict] = []
    options.extend([{
            "label": "pick up the container (click segmentation)",
            "solve": lambda require_target=require_target: solve_pickup_bin(
                env, planner, obj=require_target()
            ),
        },
        {
            "label": "put down the container",
            "solve": lambda require_target=require_target: solve_putdown_whenhold(
                env, planner, obj=require_target(), release_z=0.01
            ),
        },])
    return options

OPTION_BUILDERS: Dict[str, Callable] = {
    "VideoRepick": _options_videorepick,
    "BinFill": _options_binfill,
    "ButtonUnmask": _options_button_unmask,
    "ButtonUnmaskSwap": _options_button_unmask_swap,
    "InsertPeg": _options_insertpeg,
    "MoveCube": _options_movecube,
    "PatternLock": _options_patternlock,
    "PickHighlight": _options_pickhighlight,
    "PickXtimes": _options_pickxtimes,
    "RouteStick": _options_routestick,
    "SwingXtimes": _options_swingxtimes,
    "StopCube": _options_stopcube,
    "VideoPlaceButton": _options_videoplacebutton,
    "VideoPlaceOrder": _options_videoplaceorder,
    "VideoUnmask": _options_video_unmask,
    "VideoUnmaskSwap": _options_video_unmask_swap,
    
}


def get_vqa_options(env, planner, selected_target, env_id: str) -> List[dict]:
    """
    根据 env_id 返回固定的一组 options（label + solve）。
    目标对象由 GUI 点击后在 selected_target 中提供。
    """
    def _require_target():
        obj = selected_target.get("obj")
        if obj is None:
            raise ValueError("未找到可用的 target cube，请先在分割图中点击目标。")
        return obj

    base = env.unwrapped
    builder = OPTION_BUILDERS.get(env_id, _options_default)
    return builder(env, planner, _require_target, base)
