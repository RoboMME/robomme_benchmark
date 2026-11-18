"""Seeded RGB multiplicities and physical difficulty presets for four Unmask tasks."""

import hashlib
import math
from types import SimpleNamespace

import numpy as np

from robomme_plus.common.sim import position
from robomme_plus.scoring.failures import record_failure
from robomme_plus.tasks.permanence.utils import support as base
from robomme_plus.tasks.permanence.utils.progress import ColorPickupProgress, cover_exposure

TASKS = ("VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap")

COLORS = ("red", "blue", "green")

RGBA = {"red": [1, 0, 0, 1], "blue": [0, 0, 1, 1], "green": [0, 1, 0, 1]}

PALETTE = [
    ("yellow", [1, 1, 0, 1]),
    ("orange", [1, 0.45, 0, 1]),
    ("violet", [0.55, 0.1, 0.85, 1]),
    ("white", [0.92, 0.92, 0.92, 1]),
    ("black", [0.04, 0.04, 0.04, 1]),
]

PRESETS = {
    "easy": dict(
        blocking=False,
        containers=dict(zip(TASKS, [(5, 10), (6, 8), (6, 8), (6, 8)])),
        empty_fraction=dict.fromkeys(TASKS, 0),
    ),
    "medium": dict(
        blocking=True,
        containers=dict(zip(TASKS, [(11, 15), (8, 12), (8, 10), (9, 10)])),
        empty_fraction=dict(zip(TASKS, [4 / 16, 2 / 10, 0, 0])),
    ),
    "hard": dict(
        blocking=True,
        containers=dict(zip(TASKS, [(16, 28), (16, 18), (9, 12), (11, 13)])),
        empty_fraction=dict(zip(TASKS, [12 / 28, 4 / 18, 0, 0])),
    ),
}

BUTTON_SWAP_GOAL_COUNTS = {"easy": (3, 3), "medium": (3, 4), "hard": (3, 4)}
BUTTON_UNMASK_DEADLINE_STEP = 90
BUTTON_SWAP_DEADLINE_STEP = 180
BUTTON_SWAP_MOTION_STOP_STEP = 480
BUTTON_SWAP_REVEAL_STEPS = 10

WORKSPACE = np.array([[-0.24870598316192627, 0.16], [-0.36, 0.36]])

PROFILE = "permanence_random_rgb"

BUTTON_HARD_CENTER = np.array([[-0.10, 0.075], [-0.16, 0.16]])

# Stationary layouts (VideoUnmask, ButtonUnmask) shrink toward the workspace center
# as far as their container count allows, at this packing fraction.
STATIC_PACKING_FRACTION = 0.55
STATIC_MIN_SCALE = 0.6
VIDEO_PACKING_FRACTION = 0.75
VIDEO_MIN_SCALE = 0.5
VIDEO_CENTER = np.c_[
    WORKSPACE.mean(axis=1) - np.ptp(WORKSPACE, axis=1) / 4,
    WORKSPACE.mean(axis=1) + np.ptp(WORKSPACE, axis=1) / 4,
]


def static_workspace(count, radius, gap=0.012, *, compact=False):
    """Centered sub-rectangle of WORKSPACE with room for ``count`` containers."""
    span = WORKSPACE[:, 1] - WORKSPACE[:, 0]
    fraction = VIDEO_PACKING_FRACTION if compact else STATIC_PACKING_FRACTION
    minimum = VIDEO_MIN_SCALE if compact else STATIC_MIN_SCALE
    needed = count * (2 * radius + gap) ** 2 / fraction
    scale = float(np.clip(math.sqrt(needed / span.prod()), minimum, 1.0))
    center = WORKSPACE.mean(axis=1)
    return np.c_[center - span * scale / 2, center + span * scale / 2], scale


BOARD_SIZE = (0.03, 0.045, 0.3)
# With boards (medium, hard), this many targets are sampled where a board hides
# them from the front camera.
BOARD_HIDDEN_TARGETS = 2


def ray_hits_box(eye, point, center, size):
    """Whether the segment from ``eye`` to ``point`` crosses an axis-aligned box."""
    eye, direction = np.asarray(eye, float), np.asarray(point, float) - eye
    low, high = (
        np.asarray(center) - np.asarray(size) / 2,
        np.asarray(center) + np.asarray(size) / 2,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        near, far = (low - eye) / direction, (high - eye) / direction
    entry = max(0.0, float(np.nanmax(np.minimum(near, far))))
    leave = min(1.0, float(np.nanmin(np.maximum(near, far))))
    return entry <= leave


def board_shadow(env, config, cube_z):
    """Predicate for table points whose cube a board hides from the front camera.

    ButtonUnmask's board also covers its mirrored position after the press.
    """
    centers = [np.asarray(c, float) for c in board_centers(config)]
    if config["task"].startswith("Button"):
        centers += [c * [1, -1, 1] for c in centers]
    if not centers:
        return None
    camera = next(c for c in env._default_sensor_configs if c.uid == "base_camera")
    eye = camera.pose.p
    eye = np.asarray(
        eye.detach().cpu().numpy() if hasattr(eye, "detach") else eye, float
    ).reshape(3)
    return lambda xy: any(
        ray_hits_box(eye, [*xy, cube_z], c, BOARD_SIZE) for c in centers
    )


def cover_quaternion(yaw):
    """Upside-down hollow cover rotated by ``yaw`` about the vertical."""
    from transforms3d.euler import euler2quat

    return np.asarray(euler2quat(math.pi, 0, yaw, axes="rxyz"))


def grasp_quaternion(yaw):
    """Top-down grasp of a square cover with yaw ``yaw`` (90-degree symmetric)."""
    from transforms3d.quaternions import axangle2quat, qmult

    relative = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return tuple(
        float(v) for v in qmult(axangle2quat([0, 0, 1], relative), [0, 1, 0, 0])
    )


def sample_goal(seed, task, *, count_range=(3, 5)):
    """Uniform inclusive total; each extra color and the group order are uniform."""
    if task not in TASKS:
        raise ValueError(task)
    digest = hashlib.sha256(f"permanence-rgb-v1/{task}/{int(seed)}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    total = int(rng.integers(count_range[0], count_range[1] + 1))
    counts = dict.fromkeys(COLORS, 1)
    extras = rng.choice(COLORS, size=total - 3).tolist()
    for color in extras:
        counts[color] += 1
    order = rng.permutation(COLORS).tolist()
    return dict(
        seed=int(seed),
        task=task,
        total_targets=total,
        target_counts=counts,
        color_order=order,
        extra_colors=extras,
        pickup_sequence=[c for c in order for _ in range(counts[c])],
    )


def language_goal(recipe):
    clauses = [
        f"the one hiding the {c} cube"
        if recipe["target_counts"][c] == 1
        else f"all containers hiding {c} cubes"
        for c in recipe["color_order"]
    ]
    clauses[0] = clauses[0].replace("the one hiding", "the container hiding", 1)
    task = recipe["task"]
    prefix = (
        "Watch the video carefully"
        if task.startswith("Video")
        else "First press both buttons once"
        if task.endswith("Swap")
        else "First press the button"
    )
    return (
        prefix
        + ", then pick up "
        + clauses[0]
        + ", followed by "
        + clauses[1]
        + ", and finally "
        + clauses[2]
        + "."
    )


def initial_board_side(task, seed):
    """Sample a fixed mirrored side independently of targets and layout retries."""
    if not task.endswith("Swap"):
        return 1
    digest = hashlib.sha256(f"permanence-board-v1/{task}/{int(seed)}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return int(rng.choice([-1, 1]))


def board_centers(config):
    if not config["blocking"]:
        return []
    if config["task"] == "VideoUnmask":
        return [[0.2, 0.045, 0.15], [0.2, -0.045, 0.15]]
    return [[config.get('board_x_m', 0.2), 0.025 * config["board_start_side"], 0.15]]


def task_config(task, difficulty, seed):
    if difficulty not in PRESETS:
        raise ValueError(difficulty)
    preset = PRESETS[difficulty]
    goal_range = BUTTON_SWAP_GOAL_COUNTS[difficulty] if task == "ButtonUnmaskSwap" else (3, 5)
    recipe = sample_goal(seed, task, count_range=goal_range)
    swap = task.endswith("Swap")
    low, high = preset["containers"][task]
    digest = hashlib.sha256(
        f"permanence-containers-v1/{task}/{difficulty}/{int(seed)}".encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    count = int(rng.integers(low, high + 1))
    # Preserve the old empty-cover proportions as the total varies. All target
    # cubes must still fit, including episodes at the bottom of a count range.
    empty = min(
        round(count * preset["empty_fraction"][task]), count - recipe["total_targets"]
    )
    passage_count = None
    if task == "ButtonUnmaskSwap":
        passage_count = 2
        if difficulty == "hard":
            passage_digest = hashlib.sha256(f"button-swap-passages-v1/{int(seed)}".encode()).digest()
            passage_rng = np.random.default_rng(int.from_bytes(passage_digest[:8], "little"))
            passage_count = int(passage_rng.integers(2, 4))
    return dict(
        task=task,
        difficulty_preset=difficulty,
        recipe=recipe,
        scenario_version=3,
        min_containers=count,
        empty_containers=empty,
        container_count_range=[low, high],
        goal_count_range=list(goal_range),
        container_count_sampling="uniform integer, inclusive endpoints, seeded independently of goal",
        blocking=preset["blocking"],
        fill_distractor_covers=True,
        occluder="board" if preset["blocking"] else None,
        board_start_side=initial_board_side(task, seed),
        board_side_sampling="uniform mirrored sides, fixed per episode"
        if swap
        else "fixed",
        extra_central_distractors=2
        if task == "ButtonUnmask" and difficulty == "hard"
        else 0,
        speed=0.00675 if task == "ButtonUnmaskSwap" and difficulty == "hard" else 0.0045,
        reveal_steps=BUTTON_SWAP_REVEAL_STEPS if task == "ButtonUnmaskSwap" else 36,
        convergence_steps=0,
        button_passage_count=passage_count,
        moving_steps=BUTTON_SWAP_MOTION_STOP_STEP - BUTTON_SWAP_REVEAL_STEPS
        if task == "ButtonUnmaskSwap" else 720 if swap else 0,
        motion_tail_steps=60,
        required_button_presses=2
        if task == "ButtonUnmaskSwap"
        else 1
        if task == "ButtonUnmask"
        else 0,
        button_deadline_steps=BUTTON_UNMASK_DEADLINE_STEP
        if task == "ButtonUnmask" else BUTTON_SWAP_DEADLINE_STEP,
        post_button_motion_steps=96,
        batch_exposure_steps=32,
        batch_gap_steps=12,
        initial_delay_steps=12,
        target_cubes_per_batch=2 if task == "ButtonUnmask" else 1,
        target_sampling_policy="same reachable continuous pool for every RGB color",
        total_sampling=f"uniform integer from {goal_range[0]} through {goal_range[1]} inclusive",
        extra_color_sampling="independent uniform red/blue/green draws",
        goal_order_sampling="uniform permutation of the three color groups",
    )


def add_objects(env, config):
    from robomme_plus.simulation.actors import spawn_fixed_cube

    count = config["min_containers"]
    # The grasp block stays above the roof so lifting covers leaves cubes still.
    env._permanence_actors = []
    while len(env._permanence_actors) < count:
        i = len(env._permanence_actors)
        env._permanence_actors.append(
            base.build_hollow_cover(
                env, callsign=f"permanence_cover_{i}", position=[30 + i, 30]
            )
        )
    filled = count - config["empty_containers"]
    recipe = config["recipe"]
    colors = list(recipe["pickup_sequence"])
    colors += [PALETTE[i % len(PALETTE)][0] for i in range(filled - len(colors))]
    palette = dict(PALETTE, **RGBA)
    cube_yaws = np.random.default_rng(recipe["seed"] + 1734003).uniform(
        -math.pi, math.pi, len(colors)
    )
    env._permanence_cubes = [
        spawn_fixed_cube(
            env,
            position=[40 + i, 40],
            half_size=env.cube_half_size / 1.2,
            color=palette[color],
            name_prefix=f"permanence_cube_{i}_{color}",
            dynamic=True,
            yaw=float(cube_yaws[i]),
        )
        for i, color in enumerate(colors)
    ]
    env._permanence_colors = colors


def add_boards(env):
    import sapien

    config = env._limit_config
    centers = board_centers(config)
    size = [0.03, 0.045, 0.3]
    env._permanence_boards = []
    for i, center in enumerate(centers):
        builder = env.scene.create_actor_builder()
        material = sapien.render.RenderMaterial(
            base_color=[0.52, 0.58, 0.66, 1], roughness=0.9
        )
        builder.add_box_visual(sapien.Pose(), np.asarray(size) / 2, material=material)
        builder.add_box_collision(sapien.Pose(), np.asarray(size) / 2)
        builder.initial_pose = sapien.Pose(center)
        env._permanence_boards.append(builder.build_kinematic(f"permanence_board_{i}"))
    env._limit_occluder = env._permanence_boards[0]
    env._limit_occluder_pose = sapien.Pose(centers[0])
    return dict(
        board_count=len(centers),
        center_xyz_m=centers[0],
        size_xyz_m=size,
        board_descriptions=[dict(center_xyz_m=p, size_xyz_m=size) for p in centers],
        initial_side=config["board_start_side"],
        side_sampling=config["board_side_sampling"],
        physical_collision=True,
        always_present=True,
        removed=False,
    )


def pixels(env, cubes=None):
    env._get_obs_sensor_data()
    cubes = (
        env._permanence_cubes[: env._limit_config["recipe"]["total_targets"]]
        if cubes is None
        else cubes
    )
    result = {}
    for view, sensor in [("front", "base_camera"), ("wrist", "hand_camera")]:
        labels = (
            env._sensors[sensor]
            .get_obs(rgb=False, depth=False, position=False, segmentation=True)[
                "segmentation"
            ]
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )
        result[view] = [
            int(np.count_nonzero(labels == int(cube.per_scene_id.item())))
            for cube in cubes
        ]
    return result


def reveal_batches(seed, target_count, container_count, targets_per_batch=1):
    """Group targets without shortening their individual visible intervals."""
    rng = np.random.default_rng(seed + 738111)
    target_batches = np.arange(target_count) // targets_per_batch
    batch_count = int(target_batches[-1]) + 1
    assignment = np.r_[
        rng.permutation(target_batches),
        np.arange(container_count - target_count) % batch_count,
    ]
    assignment[target_count:] = rng.permutation(assignment[target_count:])
    return assignment.tolist(), batch_count


def button_cap(env, button):
    """The button's cap link, which upstream RoboMME grounds for press subgoals."""
    for links in getattr(env, "cap_links", {}).values():
        if any(link is mine for link in links for mine in button.links):
            return links
    return button


def buttons(env):
    # Swap exposes button as an alias of button_left; count physical buttons once.
    unique = {}
    for name in ["button", "button_left", "button_right"]:
        if hasattr(env, name):
            actor = getattr(env, name)
            unique.setdefault(actor.name, actor)
    return list(unique.values())


def button_subgoals(env):
    """Use RoboMME's press labels; first/second describe chronological presses."""
    pressable = buttons(env)
    if len(pressable) == 2:
        def caps(selected):
            objects = []
            for button in selected:
                cap = button_cap(env, button)
                objects.extend(cap if isinstance(cap, (list, tuple)) else [cap])
            return objects

        def remaining_caps():
            return caps([b for b in pressable if b.name not in env._permanence_presses])

        def first_cap():
            # Recommend RoboMME's first oracle button, but accept either order.
            # On the completion frame, ground the button actually pressed first.
            name = next(iter(env._permanence_presses), env.button_right.name)
            return button_cap(env, next(b for b in pressable if b.name == name))

        return [
            base.task("press the first button", lambda: len(env._permanence_presses) >= 1,
                      None, segment=first_cap,
                      grounded="press the first button at <>"),
            base.task("press the second button", lambda: len(env._permanence_presses) == 2,
                      None, segment=remaining_caps, grounded="press the second button at <>"),
        ]
    return [base.task("press the button", lambda name=b.name: name in env._permanence_presses,
                      None, segment=button_cap(env, b), grounded="press the button at <>")
            for b in pressable]


def static_layout(env, config, reach, radius, z):
    from robomme_plus.tasks.permanence.utils.sampling import (
        packed_points,
        sample_target_positions,
    )

    recipe = config["recipe"]
    rng = np.random.default_rng(recipe["seed"] + 1734001)
    obstacles = [
        (position(button, initial=False)[:2], 0.055) for button in buttons(env)
    ]

    def allowed(xy):
        if any(np.linalg.norm(xy - p) <= radius + r + 0.012 for p, r in obstacles):
            return False
        for board in getattr(env, "_permanence_boards", []):
            center = position(board, initial=False)[:2].copy()
            extent = np.array([0.015, 0.0225])
            if config["task"].startswith("Button"):
                extent[1] += abs(center[1])
                center[1] = 0
            if (
                np.linalg.norm(np.maximum(np.abs(xy - center) - extent, 0))
                <= radius + 0.012
            ):
                return False
        return True

    n = recipe["total_targets"]
    compact = config["task"] == "VideoUnmask"
    bounds, scale = static_workspace(
        len(env._permanence_actors), radius, compact=compact
    )
    hidden_bounds = (
        np.c_[
            np.maximum(bounds[:, 0], VIDEO_CENTER[:, 0]),
            np.minimum(bounds[:, 1], VIDEO_CENTER[:, 1]),
        ]
        if compact
        else bounds
    )
    shadow = board_shadow(env, config, env.cube_half_size / 1.2)
    hidden_count = min(BOARD_HIDDEN_TARGETS, n) if shadow else 0
    gap = 2 * radius + 0.012
    # Every container gets a random yaw; covers are square, so [0, 90) degrees.
    yaws = np.random.default_rng(recipe["seed"] + 1734002).uniform(
        0, math.pi / 2, len(env._permanence_actors)
    )
    for attempt in range(150):
        try:
            hidden = (
                sample_target_positions(
                    rng,
                    hidden_count,
                    hidden_bounds,
                    radius,
                    lambda xy: shadow(xy) and reach.check(xy, z),
                    allowed,
                )
                if hidden_count
                else np.empty((0, 2))
            )
            central, _ = packed_points(
                rng,
                config["extra_central_distractors"],
                BUTTON_HARD_CENTER,
                radius,
                hidden,
                allowed,
                accept=lambda xy: reach.check(xy, z),
                gap=0.012,
            )
            rest, _ = packed_points(
                rng,
                len(env._permanence_actors) - len(hidden) - len(central),
                bounds,
                radius,
                np.r_[hidden, central],
                allowed,
                accept=lambda xy: reach.check(xy, z),
                gap=0.012,
            )
        except RuntimeError:
            continue
        # Target and distractor roles share the same packed pool and feasibility
        # checks. Reserve only the board-hidden goals required by the task.
        targets = np.r_[hidden, rest[:n - hidden_count]][rng.permutation(n)]
        points = np.r_[targets, central, rest[n - hidden_count:]]
        if not all(reach.check(xy, z, grasp_quaternion(yaw))
                   for xy, yaw in zip(points, yaws)):
            continue
        for i, cube in enumerate(env._permanence_cubes):
            base.set_position(cube, [*points[i], env.cube_half_size / 1.2])
        counts = pixels(env)
        if all(max(a, b) >= 12 for a, b in zip(counts["front"], counts["wrist"])):
            env._limit_diagnostics["central_distractors"] = dict(
                count=len(central),
                bounds_xy_m=BUTTON_HARD_CENTER.tolist(),
                cube_colors=env._permanence_colors[n : n + len(central)],
                positions_xy_m=central.tolist(),
                each_has_a_container=True,
            )
            env._limit_diagnostics["static_layout"] = dict(
                bounds_xy_m=bounds.tolist(),
                workspace_scale=scale,
                board_hidden_targets=int(sum(shadow(xy) for xy in targets))
                if shadow
                else 0,
                required_board_hidden_targets=hidden_count,
                hidden_target_bounds_xy_m=hidden_bounds.tolist(),
                central_board_hidden_targets=int(
                    sum(
                        np.all(xy >= VIDEO_CENTER[:, 0])
                        and np.all(xy <= VIDEO_CENTER[:, 1])
                        and shadow(xy)
                        for xy in targets
                    )
                )
                if compact and shadow
                else 0,
                required_central_board_hidden_targets=hidden_count if compact else 0,
                packing_fraction=VIDEO_PACKING_FRACTION
                if compact
                else STATIC_PACKING_FRACTION,
                container_yaw_rad=yaws.tolist(),
            )
            return points, counts, attempt + 1, [cover_quaternion(yaw) for yaw in yaws]
    raise RuntimeError("Cannot pack visible, reachable Permanence targets")


def swap_routes(env, config, reach, radius, z):
    from robomme_plus.tasks.permanence.utils import moving as moving
    from robomme_plus.tasks.permanence.utils import routes as flow
    from robomme_plus.tasks.permanence.utils.reachability import ReachableArea

    area = ReachableArea(
        env, z, max_x=(config.get('motion_workspace') or {'bounds_xy_m': flow.BOUNDS})['bounds_xy_m'][0][1] - radius
        if config["task"] == "ButtonUnmaskSwap" else 0.14,
    )
    # A circle encloses the board at both mirrored positions and throughout
    # its sweep. Expanded board-side routes must clear it at every step.
    board_obstacles = [
        ([center[0], 0.0], math.hypot(BOARD_SIZE[0] / 2,
                                    abs(center[1]) + BOARD_SIZE[1] / 2))
        for center in board_centers(config)
    ] if config["task"] == "ButtonUnmaskSwap" else []
    if config["task"] == "ButtonUnmaskSwap":
        # Sample the buttons jointly with their required robot-side crossing,
        # before packing covers. A valid button pair must leave the entire
        # crossing reachable, rather than failing later for particular seeds.
        rng = np.random.default_rng(config["recipe"]["seed"] + 1734004)
        for attempt in range(4000):
            pair = np.array([rng.uniform([-0.25, -0.15], [-0.14, -0.06]),
                             rng.uniform([-0.25, 0.06], [-0.14, 0.15])])
            proposed = [(p, 0.055) for p in pair]
            try:
                rear, front = flow.plan_routes(len(env._permanence_actors), radius,
                    config["recipe"]["seed"] + 742931, speed=config["speed"],
                    obstacles=proposed, reachable_area=area, geometry_only=True,
                    continuous=config.get('continuous_motion', False),
                    workspace=config.get('motion_workspace'),
                    extra_obstacles=board_obstacles,
                    initial_board_front_x=config.get('board_x_m', 0.2) - BOARD_SIZE[0] / 2)
            except ValueError:
                continue
            for button, point in zip((env.button_right, env.button_left), pair):
                p = position(button, initial=False).copy()
                p[:2] = point
                base.set_position(button, p)
            env._limit_diagnostics["button_layout"] = dict(
                positions_xy_m=pair.tolist(), attempts=attempt + 1,
                robot_side_crossing_xy_m=rear.tolist(),
                sampling="joint uniform proposals conditioned on a reachable crossing and initial packing between buttons and board")
            break
        else:
            raise RuntimeError("Cannot sample buttons with a reachable robot-side crossing")
    obstacles = [(position(b, initial=False)[:2], 0.055) for b in buttons(env)]
    strategy = flow if config["task"] == "ButtonUnmaskSwap" else moving

    def plan(count, radius, _background_seed, **kwargs):
        if config["task"] == "ButtonUnmaskSwap":
            kwargs["extra_obstacles"] = board_obstacles
            if kwargs.get('pickup_passing'):
                # Preserve the board's full sweep envelope without replacing
                # its thin tabletop footprint by an oversized circle.
                half_x = BOARD_SIZE[0]/2
                kwargs['rectangle_obstacles'] = [
                    (p, [half_x, math.sqrt(r*r-half_x*half_x)]) for p, r in board_obstacles]
                kwargs['extra_obstacles'] = ()
            kwargs["passage_count"] = config["button_passage_count"]
            kwargs["initial_board_front_x"] = config.get('board_x_m', 0.2) - BOARD_SIZE[0] / 2
            kwargs['workspace'] = config.get('motion_workspace')
        route_planner = strategy.plan_routes
        return route_planner(
            count,
            radius,
            config["recipe"]["seed"] + 742931,
            reachable_area=area,
            **kwargs,
        )

    route_class = base.Routes
    if config.get("continuous_motion"):
        if config.get('continuous_motion_version', 1) == 1:
            from .continuous_v1 import ContinuousRoutes
        elif config['continuous_motion_version'] == 2:
            from .continuous_v2 import ContinuousRoutes
        elif config['continuous_motion_version'] == 3:
            from .continuous_v3 import ContinuousRoutes
        elif config['continuous_motion_version'] == 4:
            from .continuous_v4 import ContinuousRoutes
        elif config['continuous_motion_version'] == 5:
            from .continuous_v5 import ContinuousRoutes
        else:
            from .continuous import ContinuousRoutes
        route_class = ContinuousRoutes
    routes = route_class(
        env,
        env._permanence_actors,
        radius,
        config["speed"],
        steps=config["moving_steps"] if config["task"] == "ButtonUnmaskSwap"
        else 36 + config["moving_steps"] + 100,
        obstacles=obstacles,
        planner=plan,
    )
    yaw_rng = np.random.default_rng(config["recipe"]["seed"] + 1734002)
    yaws = []
    for index, height in enumerate(routes.z):
        if config.get("continuous_motion"):
            # The online traffic region is validated for a straight top grasp.
            yaws.append(0.0)
            continue
        path = routes.paths[:, index]
        checkpoints = np.r_[path, (path[:-1] + path[1:]) / 2]
        for _ in range(64):
            yaw = float(yaw_rng.uniform(0, math.pi / 2))
            if all(area.reach.check(xy, height, grasp_quaternion(yaw)) for xy in checkpoints):
                yaws.append(yaw)
                break
        else:
            raise RuntimeError("Cannot find a reachable cover orientation along its full route")
    routes.q = [cover_quaternion(yaw) for yaw in yaws]
    routes.metadata["container_yaw_rad"] = yaws
    audit = area.document(routes.paths)
    if config.get("continuous_motion"):
        audit.update(all_container_routes_checked=False, exact_ik_after_route_generation=False,
                     guarantee="continuous traffic within the top-down grasp reachability grid")
        if config['continuous_motion_version'] >= 4:
            audit.update(rotation_aligned_grasps_prevalidated=False,
                         guarantee='continuous traffic within the straight top-down grasp center grid; rotating cover-aligned grasp orientations are not prevalidated')
    audit["randomized_cover_grasps_checked_along_entire_route"] = not env._limit_config.get('continuous_motion', False)
    for i, actor in enumerate(routes.actors):
        base.set_position(actor, [30 + i, 30, 2], routes.q[i])
    rng = np.random.default_rng(config["recipe"]["seed"] + 172921)
    n = config["recipe"]["total_targets"]
    for attempt in range(100):
        permutation = rng.permutation(len(routes.actors))
        for i, cube in enumerate(env._permanence_cubes):
            base.set_position(
                cube, [*routes.paths[0, permutation[i]], env.cube_half_size / 1.2]
            )
        counts = pixels(env)
        if all(max(a, b) >= 12 for a, b in zip(counts["front"], counts["wrist"])):
            env._limit_diagnostics["reachable_container_motion"] = audit
            return routes, permutation, counts
    raise RuntimeError(f"Cannot reveal all {n} moving target cubes")


def setup(env, config):
    from robomme_plus.scoring.predicates import is_bin_pickup, is_bin_putdown
    from robomme_plus.tasks.permanence.utils.board import (
        ButtonBoard,
        ButtonDeadline,
    )
    from robomme_plus.tasks.permanence.utils.reachability import (
        CoverReachability,
    )
    from robomme_plus.tasks.permanence.utils.view import initialize_view

    recipe = config["recipe"]
    for i, actor in enumerate(env._permanence_actors):
        base.set_position(
            actor,
            [
                30 + i,
                30,
                float(actor.initial_pose.p.detach().cpu().numpy().reshape(-1, 3)[0, 2]),
            ],
        )
    for board, row in zip(
        getattr(env, "_permanence_boards", []),
        env._limit_diagnostics.get("occluder", {}).get("board_descriptions", []),
    ):
        base.set_position(board, row["center_xyz_m"])
    initialize_view(env)
    reach = CoverReachability(env)
    radius = math.sqrt(2) * (env.cube_half_size * 2.5 + 0.005) / 2
    actors = env._permanence_actors
    z = [float(position(a, initial=False)[2]) for a in actors]
    quaternions = [
        a.pose.q.detach().cpu().numpy().reshape(-1, 4)[0].copy() for a in actors
    ]
    if config["task"].endswith("Swap"):
        routes, permutation, counts = swap_routes(env, config, reach, radius, z[0])
    else:
        points, counts, attempts, quaternions = static_layout(
            env, config, reach, radius, z[0]
        )
        permutation = np.arange(len(actors))
        routes = SimpleNamespace(
            actors=actors,
            paths=np.repeat(points[None], 2, axis=0),
            z=z,
            q=quaternions,
            active=False,
            execution_start=None,
            radius=radius,
            metadata=dict(
                strategy="stationary staged covers", layout_attempts=attempts
            ),
            history=[points.copy()],
        )
    env._limit_routes = routes
    env._permanence_permutation = permutation.tolist()
    env._limit_hidden = [
        (cube, int(permutation[i]), float(env.cube_half_size / 1.2))
        for i, cube in enumerate(env._permanence_cubes)
    ]
    n = recipe["total_targets"]
    env._repeat_target_count = n
    env._repeat_target_colors = env._permanence_colors[:n]
    mapping = {
        actors[int(permutation[i])].name: color
        for i, color in enumerate(env._repeat_target_colors)
    }
    progress = env._repeat_progress = ColorPickupProgress(
        recipe["color_order"], recipe["target_counts"], mapping
    )
    env._repeat_pick_events = []
    env._permanence_cover_spawned = set()
    env._permanence_cube_spawned = set()
    env._permanence_cube_released = False
    env._permanence_presses = {}
    env._permanence_early_opening = None
    env._permanence_deadline = None
    env._permanence_board = None
    if config["task"].startswith("Button"):
        env._permanence_deadline = ButtonDeadline(
            [b.name for b in buttons(env)],
            config["reveal_steps"] if config["task"] == "ButtonUnmaskSwap" else 0,
            config["button_deadline_steps"], config["post_button_motion_steps"],
            motion_stop_step=BUTTON_SWAP_MOTION_STOP_STEP
            if config["task"] == "ButtonUnmaskSwap" else None,
            fixed_motion_stop=False,
            continuous_motion=config.get("continuous_motion", False),
            repeat_press_allowed=config["task"] == "ButtonUnmask",
        )
    if config["task"].startswith("Button") and config["blocking"]:
        origin = env._limit_diagnostics["occluder"]["center_xyz_m"]
        env._permanence_board = ButtonBoard(origin, [origin[0], -origin[1], origin[2]])
    # ButtonUnmask reveals target pairs; other cubes/covers arrive in the same
    # batches. Every cube still has its full 32-step exposure before covering.
    assignment, batch_count = reveal_batches(
        recipe["seed"], n, len(actors), config["target_cubes_per_batch"]
    )
    env._permanence_batch_assignment = assignment
    env._permanence_batch_start = int(base.step_number(env))
    env._permanence_total_reveal = 12 + 44 * batch_count
    env._permanence_events = []
    env._permanence_cube_writes_after_release = 0
    env._limit_reveal_start = int(base.step_number(env))
    env._limit_reveal_steps = config["reveal_steps"]
    env._limit_motion_steps = config["moving_steps"]
    env._limit_setup = True
    env._home_return_done = False

    def raised():
        return [
            a.name
            for a in actors
            if a.name in env._permanence_cover_spawned
            and base.scalar(is_bin_pickup(env, obj=a))
        ]

    def accept(stage):
        before = len(progress.picked)
        value = progress.observe(stage, raised())
        if len(progress.picked) > before:
            env._repeat_pick_events.append(
                dict(
                    actor=progress.picked[-1],
                    color=progress.sequence[stage],
                    stage=stage,
                    control_step=base.step_number(env),
                )
            )
        return value

    def invalid_pick(stage):
        reason = progress.invalid_reason(stage, raised())
        if reason is not None:
            record_failure(env, reason)
        return reason is not None

    def target_container(stage):
        # Scoring accepts the pickup before grounding its completion frame.
        if len(progress.picked) > stage:
            return next(a for a in actors if a.name == progress.picked[stage])
        # Same-color containers are interchangeable; ground the one the oracle picks.
        eligible = [a for a in actors if a.name in progress.allowed(stage)]
        tcp = position(env.agent.tcp, initial=False)[:2]
        return (
            min(
                eligible,
                key=lambda a: np.linalg.norm(position(a, initial=False)[:2] - tcp),
            )
            if eligible
            else None
        )

    # Subgoal names and grounded templates follow upstream RoboMME.
    video = config["task"].startswith("Video")
    tasks = button_subgoals(env)
    if not config.get("continuous_motion"):
        tasks.append(
            base.task(
                "static" if video else "remain static",
                lambda: motion_finished(env),
                lambda e, p: wait(e, p),
                demo=video,
            )
        )
    if config["task"] == "VideoUnmask":
        tasks.append(
            base.task(
                "NO RECORD", lambda: env._home_return_done, return_home, demo=True
            )
        )
    seen = {}
    for stage, color in enumerate(progress.sequence):
        if stage:
            tasks.append(
                base.task(
                    "put down the container",
                    lambda s=stage: (
                        len(progress.picked) >= s
                        and base.scalar(
                            is_bin_putdown(
                                env,
                                obj=next(
                                    a
                                    for a in actors
                                    if a.name == progress.picked[s - 1]
                                ),
                            )
                        )
                    ),
                    None,
                    failure=lambda s=stage: bool(
                        set(raised()) - {progress.picked[s - 1]}
                    ),
                    failure_reason="container_before_putdown",
                )
            )
        if recipe["target_counts"][color] == 1:
            name = f"pick up the container that hides the {color} cube"
            grounded = f"pick up the container at <> that hides the {color} cube"
        else:
            nth = ("first", "second", "third", "fourth", "fifth")[seen.get(color, 0)]
            name = f"pick up the {nth} container that hides a {color} cube"
            grounded = f"pick up the {nth} container at <> that hides a {color} cube"
        seen[color] = seen.get(color, 0) + 1
        tasks.append(
            base.task(
                name,
                lambda s=stage: accept(s),
                None,
                failure=lambda s=stage: invalid_pick(s),
                segment=lambda s=stage: target_container(s),
                grounded=grounded,
            )
        )
    env.task_list = tasks
    env._limit_diagnostics.update(
        recipe=recipe,
        task_goal=language_goal(recipe),
        challenge_seed=recipe["seed"],
        native_background_seed=int(env.seed),
        required_target_count=n,
        target_counts=recipe["target_counts"],
        color_order=recipe["color_order"],
        reveal_batch_count=batch_count,
        target_cubes_per_batch=config["target_cubes_per_batch"],
        cube_exposure_control_steps=config["batch_exposure_steps"],
        total_staged_reveal_control_steps=env._permanence_total_reveal,
        scored_pickup_sequence=progress.sequence,
        total_container_count=len(actors),
        empty_container_count=config["empty_containers"],
        blocking=config["blocking"],
        board_count=len(getattr(env, "_permanence_boards", [])),
        target_reachability=reach.document(),
        rendered_target_pixels=counts,
        all_targets_initially_visible_in_at_least_one_camera=True,
        required_button_subgoals=[b.name for b in buttons(env)],
        batch_assignment=env._permanence_batch_assignment,
        batch_events=env._permanence_events,
        repeat_pick_events=env._repeat_pick_events,
        pickup_center_height_threshold_m=0.15,
        same_color_covers_interchangeable=True,
        task_goal_has_three_color_groups=True,
        api_calls=0,
    )
    for i, actor in enumerate(actors):
        base.set_position(actor, [30 + i, 30, 2], quaternions[i])
    for i, cube in enumerate(env._permanence_cubes):
        base.set_position(cube, [40 + i, 40, 2])


def motion_finished(env):
    config = env._limit_config
    now = base.step_number(env)
    if not config["task"].endswith("Swap"):
        return now >= env._permanence_batch_start + env._permanence_total_reveal
    if env._permanence_deadline is not None:
        if config.get("continuous_motion"):
            return False
        return now >= env._permanence_deadline.motion_stop
    return now >= env._limit_reveal_start + 36 + config["moving_steps"]


def wait(env, planner):
    native = env.unwrapped
    while not motion_finished(native):
        planner.open_gripper(t=1)


def return_home(env, planner):
    from robomme_plus.tasks.permanence.utils.view import (
        return_home_with_board,
    )

    return_home_with_board(env, planner)


def observe_buttons(env):
    now = base.step_number(env)
    depths = {b.name: float(-b.get_qpos()[0, 0].item()) for b in buttons(env)}
    deadline = env._permanence_deadline
    if deadline is not None:
        deadline.observe(
            now, depths, deadline.start <= now and (deadline.motion_stop is None or now < deadline.motion_stop),
            tcp_position=position(env.agent.tcp, initial=False),
        )
        env._permanence_presses = dict(deadline.presses)
        if deadline.failed:
            record_failure(env, deadline.failure_reason)
            env._limit_failed = True
        env._limit_diagnostics["button_deadline"] = deadline.snapshot()
    else:
        for name, depth in depths.items():
            if depth > 0.005:
                env._permanence_presses.setdefault(name, now)
    board = env._permanence_board
    if board is not None:
        board.observe(now, env._permanence_presses)
        base.set_position(env._limit_occluder, board.advance(now))
        env._limit_diagnostics["button_board"] = dict(
            events=board.events,
            origin_xyz_m=board.origin.tolist(),
            mirror_xyz_m=board.side.tolist(),
            position_xyz_m=board.position.tolist(),
            always_present=True,
            move_control_steps=32,
        )


def check_early_opening(env):
    """Latch early exposure before a scripted route can restore a knocked cover."""
    if (env._limit_config["task"] != "ButtonUnmaskSwap"
            or not env._permanence_cover_spawned
            or env._limit_failed
            or len(env._permanence_presses) == 2):
        return
    routes = env._limit_routes
    for cube, index, z in env._limit_hidden:
        cover = routes.actors[index]
        if cover.name not in env._permanence_cover_spawned:
            continue
        cover_xyz = position(cover, initial=False)
        cube_xyz = position(cube, initial=False)
        parked = np.max(np.abs(cube_xyz[:2])) > 10
        if parked:
            # Moving cubes are kept off-table while masked. Their intended
            # volume follows the last commanded route point, not a displaced
            # cover's actual pose; otherwise a lateral knock would be missed.
            cube_xyz = np.array([*routes.location(index), z])
        cover_q = cover.pose.q.detach().cpu().numpy().reshape(-1, 4)[0]
        cube_q = cube.pose.q.detach().cpu().numpy().reshape(-1, 4)[0]
        reason = cover_exposure(cover_xyz, cover_q, cube_xyz, cube_q, env.cube_half_size)
        if reason is not None:
            record_failure(env, "early_container_opening")
            env._limit_failed = True
            env._permanence_early_opening = dict(
                reason=reason, control_step=int(base.step_number(env)),
                container=cover.name, cube=cube.name,
                accepted_buttons=sorted(env._permanence_presses),
                cover_xyz_m=cover_xyz.tolist(), cube_xyz_m=cube_xyz.tolist(),
                cube_was_masked=bool(parked),
            )
            env._limit_diagnostics["early_container_opening"] = env._permanence_early_opening
            return


def update(env):
    observe_buttons(env)
    check_early_opening(env)
    config = env._limit_config
    now = base.step_number(env)
    routes = env._limit_routes
    swap = config["task"].endswith("Swap")
    elapsed = now - env._limit_reveal_start
    reveal_steps = env._limit_reveal_steps
    if swap:
        if config["task"] == "ButtonUnmaskSwap":
            moving = (
                reveal_steps <= elapsed
                and (config.get("continuous_motion") or now < env._permanence_deadline.motion_stop)
                and not env._limit_failed
            )
        elif routes.execution_start is None:
            moving = 36 <= elapsed < 36 + config["moving_steps"]
        else:
            moving = now - routes.execution_start < 60
        routes.active = moving
        if elapsed >= reveal_steps and not env._permanence_cover_spawned:
            for i, actor in enumerate(routes.actors):
                base.set_position(
                    actor, [*routes.paths[0, i], routes.z[i]], routes.q[i]
                )
                env._permanence_cover_spawned.add(actor.name)
        if config.get("continuous_motion"):
            for cube, i, z in env._limit_hidden:
                if cube.name not in env._permanence_cube_spawned:
                    base.set_position(cube, [*routes.location(i), z])
                    env._permanence_cube_spawned.add(cube.name)
            if moving:
                routes.update()
            env._limit_diagnostics["routes"] = routes.metadata
            env._limit_diagnostics["continuous_motion"] = True
            return
        if moving:
            routes.update()
        reveal = elapsed < reveal_steps
        release = (config["task"].startswith("Button") and motion_finished(env)) or (
            config["task"].startswith("Video")
            and routes.execution_start is not None
            and not moving
        )
        for cube, i, z in env._limit_hidden:
            if reveal and cube.name not in env._permanence_cube_spawned:
                base.set_position(cube, [*routes.paths[0, i], z])
                env._permanence_cube_spawned.add(cube.name)
            elif not reveal and not env._permanence_cube_released:
                base.set_position(
                    cube,
                    [*position(routes.actors[i], initial=False)[:2], z]
                    if release
                    else [40 + i, 40, 2],
                )
        if release:
            env._permanence_cube_released = True
    else:
        elapsed = now - env._permanence_batch_start
        for j, (cube, i, z) in enumerate(env._limit_hidden):
            start = 12 + 44 * env._permanence_batch_assignment[j]
            if elapsed >= start and cube.name not in env._permanence_cube_spawned:
                base.set_position(cube, [*routes.paths[0, i], z])
                env._permanence_cube_spawned.add(cube.name)
                env._permanence_events.append(
                    dict(
                        cube=cube.name,
                        phase="reveal",
                        control_step=now,
                        batch=env._permanence_batch_assignment[j],
                    )
                )
        for j, i in enumerate(env._permanence_permutation):
            start = 12 + 44 * env._permanence_batch_assignment[j]
            actor = routes.actors[i]
            if (
                elapsed >= start + 32
                and actor.name not in env._permanence_cover_spawned
            ):
                base.set_position(
                    actor, [*routes.paths[0, i], routes.z[i]], routes.q[i]
                )
                env._permanence_cover_spawned.add(actor.name)
                env._permanence_events.append(
                    dict(
                        cover=actor.name,
                        phase="cover",
                        control_step=now,
                        batch=env._permanence_batch_assignment[j],
                    )
                )
        if motion_finished(env):
            env._permanence_cube_released = True
    env._limit_diagnostics["cube_release_complete"] = env._permanence_cube_released
    env._limit_diagnostics["routes"] = routes.metadata


def evaluate(env, solve_complete_eval=False):
    observe_buttons(env)
    check_early_opening(env)
    return _BASE_EVALUATE(env, solve_complete_eval)


_BASE_EVALUATE = base.evaluate_tasks


def start_execution(env):
    routes = env.unwrapped._limit_routes
    routes.execution_start = base.step_number(env.unwrapped)
    env.unwrapped._limit_diagnostics["execution_start_control_step"] = (
        routes.execution_start
    )
