from typing import Callable, Dict, List
import numpy as np

from historybench.HistoryBench_env.util.planner import (
    grasp_and_lift_peg_side,
    insert_peg,
    solve_button,
    solve_hold_obj,
    solve_pickup,
    solve_pickup_bin,
    solve_push_to_target,
    solve_push_to_target_with_peg,
    solve_putdown_whenhold,
    solve_putonto_whenhold,
    solve_putonto_whenhold_binspecial,
    solve_strong_reset,
    solve_swingonto,
    solve_swingonto_withDirection,
)
from historybench.HistoryBench_env.util.evaluate import direction


def _options_default(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = []

    return options


def _options_videorepick(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = [
        {
            "label": "pickup (click segmentation to choose target cube)",
            "solve": lambda require_target=require_target: solve_pickup(
                env, planner, obj=require_target()
            ),
        },
        {
            "label": "putdown (selected cube)",
            "solve": lambda require_target=require_target: solve_putdown_whenhold(
                env, planner, obj=require_target(), release_z=0.01
            ),
        },
    ]
    button_obj = getattr(base, "button_left", None)
    if button_obj is not None:
        options.append(
            {
                "label": "press button",
                "solve": lambda button_obj=button_obj: solve_button(
                    env, planner, obj=button_obj
                ),
            }
        )
    return options


def _options_binfill(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = [
        {
            "label": "pickup (click segmentation to choose target cube)",
            "solve": lambda require_target=require_target: solve_pickup(
                env, planner, obj=require_target()
            ),
        },
    ]
    target = getattr(base, "board_with_hole", None)
    if target is not None:
        options.append(
            {
                "label": "put into bin (selected cube)",
                "solve": lambda require_target=require_target, target=target: solve_putonto_whenhold_binspecial(
                    env, planner, obj=require_target(), target=target
                ),
            }
        )
    button_obj = getattr(base, "button", None)
    if button_obj is not None:
        options.append(
            {
                "label": "press button",
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
                "label": "press button",
                "solve": lambda button_obj=button_obj: solve_button(
                    env, planner, obj=button_obj
                ),
            }
        )

    options.extend([{
            "label": "pickup (click segmentation to choose target cube)",
            "solve": lambda require_target=require_target: solve_pickup_bin(
                env, planner, obj=require_target()
            ),
        },
        {
            "label": "putdown (selected cube)",
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
                "label": "press left button",
                "solve": lambda button_obj=button_obj_left: solve_button(
                    env, planner, obj=button_obj
                ),
            }
        )
    button_obj_right = getattr(base, "button_right", None)
    if button_obj_right is not None:
        options.append(
            {
                "label": "press right button",
                "solve": lambda button_obj=button_obj_right: solve_button(
                    env, planner, obj=button_obj
                ),
            }
        )


    options.extend([{
            "label": "pickup (click segmentation to choose target cube)",
            "solve": lambda require_target=require_target: solve_pickup_bin(
                env, planner, obj=require_target()
            ),
        },
        {
            "label": "putdown (selected cube)",
            "solve": lambda require_target=require_target: solve_putdown_whenhold(
                env, planner, obj=require_target(), release_z=0.01
            ),
        },])
    return options 


def _options_insertpeg(env, planner, require_target, base) -> List[dict]:
    options: List[dict] = []

    options.append(
        {
            "label": "pickup (click segmentation to choose peg end)",
            "solve": lambda require_target=require_target: grasp_and_lift_peg_side(
                env, planner, obj=require_target()
            ),
        }
    )

    options.append(
        {
            "label": "Insert the peg from the right side of the box",
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
            "label": "Insert the peg from the left side of the box",#和subgoal保持一致
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
                "label": "Pickup the cube",
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
    Provide 8-direction options; infer the current swing target from env state.
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

    def _current_patternlock_target():
        buttons = list(getattr(base, "selected_buttons", []) or [])
        if not buttons:
            raise ValueError("PatternLock requires selected_buttons to be initialized.")

        t = int(getattr(base, "timestep", 0) or 0)
        num_buttons = len(buttons)
        move_count = max(num_buttons - 1, 0)

        if t <= 0:
            return buttons[0], None, True  # first target, record swing_qpos
        if 1 <= t <= move_count:
            idx = t
            return buttons[idx], buttons[idx - 1], False
        if t == num_buttons:
            return "reset_home", None, False
        if t == num_buttons + 1:
            return "reset_swing", None, False
        if num_buttons + 2 <= t <= num_buttons + 1 + move_count:
            idx = t - (num_buttons + 1)
            return buttons[idx], buttons[idx - 1], False
        return None, None, False

    def _solve_direction(chosen_dir: str):
        target, last, need_record = _current_patternlock_target()
        if target is None:
            print(f"[PatternLock] No valid target for timestep {getattr(base, 'timestep', None)}")
            return

        if target == "reset_home":
            return solve_strong_reset(env, planner, gripper="stick")

        if target == "reset_swing":
            swing_qpos = getattr(base, "swing_qpos", None)
            if swing_qpos is not None:
                return solve_strong_reset(env, planner, gripper="stick", action=swing_qpos)
            return solve_strong_reset(env, planner, gripper="stick")

        if last is not None:
            expected_dir = direction(target, last, direction=8)
            if expected_dir != chosen_dir:
                print(f"[PatternLock] Expected direction {expected_dir}, got {chosen_dir}; using expected target.")

        record_flag = bool(need_record or getattr(base, "swing_qpos", None) is None)
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
        ("left clockwise", "left", "clockwise"),
        ("right clockwise", "right", "clockwise"),
        ("left counterclockwise", "left", "counterclockwise"),
        ("right counterclockwise", "right", "counterclockwise"),
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
                "label": "pickup (click segmentation to choose target cube)",
                "solve": lambda require_target=require_target: solve_pickup(
                    env, planner, obj=require_target()
                ),
            },
            {
                "label": "putdown (selected cube)",
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
                "label": "pickup (click segmentation to choose target cube)",
                "solve": lambda require_target=require_target: solve_pickup(
                    env, planner, obj=require_target()
                ),
            },
            {
                "label": "putdown (selected cube)",
                "solve": lambda require_target=require_target: solve_putonto_whenhold(
                    env, planner, obj=require_target(), target=env.target
                ),
            },
            {
                "label": "press button",
                "solve": lambda: solve_button(
                    env, planner, obj=env.button
                ),
            }
        ]
    )

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
