"""Bin Fill: scene construction, native scoring and RoboMME-Plus task configuration."""

import copy
from typing import Dict, Union

import numpy as np
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

# Robomme
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

import robomme_plus.control.reset as reset_panda
import robomme_plus.scoring.language as subgoal_language
from robomme_plus.simulation.base import MemoryTask
from robomme_plus.simulation.difficulty import (
    is_newvalue_difficulty,
    normalize_robomme_difficulty,
)
from robomme_plus.simulation.specification import EpisodeSpecError, SpecRecorder
from robomme_plus.common.errors import SceneGenerationError
from robomme_plus.common.logging_utils import logger
from robomme_plus.simulation.actors import (
    _build_new_cube_obb2d,
    _obb2d_intersect,
    build_board_with_hole,
    build_button,
    spawn_random_cube,
)
from robomme_plus.control.planning import (
    solve_button,
    solve_pickup,
    solve_putonto_whenhold_binspecial,
)
from robomme_plus.scoring.predicates import (
    check_in_bin_number,
    is_any_obj_dropped_onto_delete,
    is_any_obj_pickup_flag_currentpickup,
    is_button_pressed,
    sequential_task_check,
)
from robomme_plus.simulation.choreography import (
    lift_and_drop_objects_back_to_original,
)
from robomme_plus.simulation.geometry import cube_obb2d_exact


# Decision values come only from the class-level config_* attributes, so there is a single source of truth.


# Only implemented modes are listed; values outside this table are rejected rather than silently falling back.
XHARD_LAYOUT_DYNAMIC = {"clutter": False}


SCENE_CONFIG = {'parameters': {'put_in_color': {'hard': [2, 3], 'xhard1': [2, 3]},
                'dynamic': {'sampler': 'torch.randint',
                            'low': 0,
                            'high_exclusive': 2,
                            'shape': [1],
                            'cast': 'bool'},
                'configs': {'hard': {'color': 3,
                                     'spawn_cubes': [10, 12],
                                     'put_in_numbers': [3, 5],
                                     'put_in_color': [2, 3]},
                            'xhard1': {'color': 3,
                                       'spawn_cubes': [12, 12],
                                       'put_in_numbers': [6, 6],
                                       'layout_mode': 'clutter',
                                       'color_mix': {'max_component': 3,
                                                     'link_m': 0.09,
                                                     'max_redraws': 64},
                                       'put_in_color': [2, 3]}}},
 'positions': {'button': {'center_xy': [-0.2, 0],
                          'randomize': True,
                          'randomize_range': [0.1, 0.4],
                          'sampling_expression': '(torch.rand(2, generator=generator) - 0.5) * '
                                                 'randomize_range',
                          'scale': 1.5,
                          'randomize_range_origin': 'Original value is the build_button parameter '
                                                    'default; the original call site did not pass '
                                                    'it, so this snapshot passes it explicitly'},
               'board': {'base_position': [0.15, 0, 0],
                         'x_offset': {'scale': 0.2, 'subtract': 0.2},
                         'y_offset': {'scale': 0.4, 'subtract': 0.2},
                         'yaw_deg': {'scale': 40, 'subtract': 20},
                         'x_expression': '0.15 + (u * 0.2 - 0.2)',
                         'y_expression': 'u * 0.4 - 0.2',
                         'yaw_expression': 'u * 40.0 - 20.0',
                         'board_side': 0.1,
                         'hole_side': 0.08,
                         'thickness': 0.05,
                         'consumer': 'build_board_with_hole only receives the position; it uses no '
                                     'randomness internally'},
               'cubes': {'region_center': [-0.1, 0],
                         'region_half_size': [0.2, 0.25],
                         'random_yaw': True,
                         'yaw_range_rad': [0, 6.283185307179586],
                         'yaw_expression': 'yaw_sample * 2 * np.pi',
                         'min_gap': 'self.cube_half_size',
                         'min_gap_value': 0.02,
                         'include_existing': False,
                         'include_goal': False,
                         'rng_per_trial': ['x', 'y', 'yaw']}},
 'decision': {'layout_mode': 'native_dynamic',
              'configs': {'hard': {'color': 3, 'spawn_cubes': [10, 12], 'put_in_numbers': [3, 5]},
                          'xhard1': {'color': 3,
                                     'spawn_cubes': [12, 12],
                                     'put_in_numbers': [6, 6],
                                     'layout_mode': 'clutter',
                                     'color_mix': {'max_component': 3,
                                                   'link_m': 0.09,
                                                   'max_redraws': 64}}}}}


def _board_strips_obb2d(board):
    """2D obstacles (pad=0) for the four board strips, formula-for-formula identical to the ``board_with_hole`` special case in ``spawn_random_cube``.

    Reads poses only and draws no random numbers; used for V5 xhard slot geometry sampling
    (no actors exist in the slot stage, so spawn helpers cannot be used to build the obstacles).
    """
    board_side = board._board_side
    hole_side = board._hole_side
    actor_pos = board.pose.p
    if isinstance(actor_pos, torch.Tensor):
        actor_pos = actor_pos[0].detach().cpu().numpy()
    board_center = np.array(actor_pos[:2], dtype=np.float64)
    board_half = board_side / 2
    hole_half = hole_side / 2
    out = []
    if board_half > hole_half:
        top_height = board_half - hole_half
        A_top = np.eye(2)
        h_top = np.array([board_half, top_height / 2])
        out.append(
            (board_center + np.array([0, hole_half + top_height / 2]), A_top, h_top)
        )
        out.append(
            (board_center + np.array([0, -(hole_half + top_height / 2)]), A_top, h_top)
        )
        left_width = board_half - hole_half
        h_left = np.array([left_width / 2, hole_half])
        out.append(
            (board_center + np.array([-(hole_half + left_width / 2), 0]), A_top, h_left)
        )
        out.append(
            (board_center + np.array([hole_half + left_width / 2, 0]), A_top, h_left)
        )
    return out


def _max_same_color_component(xy, labels, link):
    """V5: size of the largest same-color connected cluster. Pure function; draws no random numbers.

    Two same-color cubes are connected when their center distance is ``<= link`` (meters); returns the
    size of the largest same-color connected component (0 for empty input).
    ``xy`` has shape ``(n, 2)``; ``labels`` has length ``n`` (any comparable color labels).
    """
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n == 0:
        return 0
    labels = list(labels)
    if len(labels) != n:
        raise ValueError(
            f"_max_same_color_component: {n} coordinates but {len(labels)} color labels"
        )
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    seen = [False] * n
    best = 0
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack, size = [start], 0
        while stack:
            u = stack.pop()
            size += 1
            for v in range(n):
                if not seen[v] and labels[v] == labels[u] and dist[u, v] <= link:
                    seen[v] = True
                    stack.append(v)
        best = max(best, size)
    return best


def _actor_xyz(actor):
    """Read only the actor's current world position, for injection evidence; changes no state and draws no random numbers."""
    pos = actor.pose.p if hasattr(actor, "pose") else actor.get_pose().p
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    return [float(v) for v in np.asarray(pos).reshape(-1)[:3]]


def _actor_quat(actor):
    """Read only the actor's current world orientation (wxyz)."""
    quat = actor.pose.q if hasattr(actor, "pose") else actor.get_pose().q
    if isinstance(quat, torch.Tensor):
        quat = quat.detach().cpu().numpy()
    return [float(v) for v in np.asarray(quat).reshape(-1)[:4]]


def _resolve_episode_spec(spec, task):
    """Prepare this instance's own copy of the fixed episode spec (new-value injection).

    Same reasoning as :func:`_resolve_sampling_config`: gymnasium stores a reference to the kwargs dict in
    ``env.unwrapped.spec.kwargs``, and the two initializations each rebuild their working state from the spec,
    so an independent deepcopy is made here; the task only modifies this working copy and never touches the
    caller's original object.

    When ``None`` is passed (no ``--episode-specs``), returns ``None`` and every consumption point follows the
    original random path, identical to before; ``DEFAULT_PARITY`` relies on exactly this.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError("episode_spec must be a dict")
    if spec.get("task") != task:
        raise ValueError(
            f"episode_spec is the spec for {spec.get('task')} and cannot be used for {task}"
        )
    return copy.deepcopy(spec)


class _TaskScene(MemoryTask):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: Union[Panda]
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)

    # config_hard = {
    # 'color': 3,
    # 'spawn_cubes':4,
    #     "put_in_color":3,
    # }

    # config_easy = {
    #     'color': 1,
    # 'spawn_cubes':8,
    #     "put_in_color":1,
    # }

    # config_medium = {
    #     'color': 3,
    # 'spawn_cubes':4,
    #     "put_in_color":1,
    # }


    # Combine into a dictionary


    # Combine into a dictionary

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0,
        seed=0,
        Robomme_video_episode=None,
        Robomme_video_path=None,
        sampling_config=None,
        episode_spec=None,
        native_episode_spec=None,
        **kwargs,
    ):
        # Must come before any RNG call and super().__init__()
        self._sampling = copy.deepcopy(SCENE_CONFIG)
        self._episode_spec = _resolve_episode_spec(episode_spec, "BinFill")
        self._spec = SpecRecorder(
            native_episode_spec,
            "BinFill",
            {"seed": seed},
            difficulty=kwargs.get("difficulty"),
        )
        self._native_init_index = -1
        self._injection_evidence = {}
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.use_demonstrationwrapper = False
        self.demonstration_record_traj = False
        normalized_robomme_difficulty = normalize_robomme_difficulty(
            kwargs.pop("difficulty", None)
        )
        self.robomme_failure_recovery = bool(
            kwargs.pop("robomme_failure_recovery", False)
        )
        self.robomme_failure_recovery_mode = kwargs.pop(
            "robomme_failure_recovery_mode", None
        )
        if isinstance(self.robomme_failure_recovery_mode, str):
            self.robomme_failure_recovery_mode = (
                self.robomme_failure_recovery_mode.lower()
            )

        if normalized_robomme_difficulty is not None:
            self.difficulty = normalized_robomme_difficulty
        else:
            # Determine difficulty based on seed % 3
            seed_mod = seed % 3
            if seed_mod == 0:
                self.difficulty = "easy"
            elif seed_mod == 1:
                self.difficulty = "medium"
            else:  # seed_mod == 2
                self.difficulty = "hard"
        # self.difficulty = "hard"

        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]

        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        dynamic_cfg = self._sampling["parameters"]["dynamic"]
        if self._episode_spec is None and is_newvalue_difficulty(self.difficulty):
            # xhard: dynamic is fixed by layout_mode and the original randint is not drawn (shifts its RNG stream).
            layout_mode = self._sampling["parameters"]["configs"][self.difficulty][
                "layout_mode"
            ]
            self._spec.record("layout.mode", layout_mode)
            self.dynamic = bool(
                self._spec.value(
                    "layout.dynamic",
                    XHARD_LAYOUT_DYNAMIC[layout_mode],
                    decision_key=f"configs.{self.difficulty}.layout_mode",
                )
            )
            self._spec.identity.setdefault(
                "difficulty", getattr(self, "difficulty", None)
            )
        self.dynamic = bool(
            self._spec.value(
                "layout.dynamic",
                bool(
                    torch.randint(
                        dynamic_cfg["low"],
                        dynamic_cfg["high_exclusive"],
                        tuple(dynamic_cfg["shape"]),
                        generator=self.generator,
                    ).item()
                ),
            )
        )
        self._spec.identity.setdefault(
            "difficulty", getattr(self, "difficulty", None)
        )

        # Track the color order and counts used to describe the language goal.
        self.binfill_language_sequence = []

        self._scene_rng_state = self.generator.get_state()
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        camera_eye = [1, 0, 0.4]
        camera_target = [0, 0, 0.4]
        pose = sapien_utils.look_at(eye=camera_eye, target=camera_target)

        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        # Preserve constructor draws as well as the original first layout.
        self.generator.set_state(self._scene_rng_state)
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Create generator for all randomization
        generator = self.generator

        spec = self._episode_spec
        button_cfg = self._sampling["positions"]["button"]
        if spec is None:
            button_obb = build_button(
                self,
                center_xy=tuple(button_cfg["center_xy"]),
                scale=button_cfg["scale"],
                generator=generator,
                randomize=button_cfg["randomize"],
                randomize_range=tuple(button_cfg["randomize_range"]),
                recorder=self._spec,
                spec_path="layout.button_xy",
            )
        else:
            button_obb = build_button(
                self,
                center_xy=tuple(spec["layout"]["button_xy"]),
                scale=button_cfg["scale"],
                generator=generator,
                randomize=False,
                randomize_range=tuple(button_cfg["randomize_range"]),
            )
        avoid = [button_obb]

        # Create square board with square hole
        board_cfg = self._sampling["positions"]["board"]
        board_x = board_cfg["x_offset"]
        board_y = board_cfg["y_offset"]
        board_yaw = board_cfg["yaw_deg"]
        board_base = board_cfg["base_position"]
        if spec is None:
            x_var = (
                torch.rand(1, generator=generator).item() * board_x["scale"]
                - board_x["subtract"]
            )  # [-0.25, 0.25]
            y_var = (
                torch.rand(1, generator=generator).item() * board_y["scale"]
                - board_y["subtract"]
            )  # [-0.25, 0.25]
            z_rot_deg = (
                torch.rand(1, generator=generator).item() * board_yaw["scale"]
                - board_yaw["subtract"]
            )  # [-20, 20] degrees
            x_var, y_var, z_rot_deg = self._spec.value(
                "layout.board.offsets", [x_var, y_var, z_rot_deg]
            )
        else:
            board_spec = spec["layout"]["board"]
            x_var = float(board_spec["xy"][0]) - float(board_base[0])
            y_var = float(board_spec["xy"][1]) - float(board_base[1])
            z_rot_deg = float(board_spec["yaw_deg"])
        z_rot_rad = torch.deg2rad(torch.tensor(z_rot_deg))
        # Create rotation quaternion for z-axis rotation
        rot_mat = euler_angles_to_matrix(
            torch.tensor([[0.0, 0.0, z_rot_rad]]), convention="XYZ"
        )
        rot_quat = matrix_to_quaternion(rot_mat)[0]  # [w, x, y, z]
        self.board_with_hole = build_board_with_hole(
            self,
            board_side=board_cfg["board_side"],  # Side length of square board
            hole_side=board_cfg[
                "hole_side"
            ],  # Side length of square hole, slightly larger than cube for passing
            thickness=board_cfg["thickness"],  # Board thickness
            position=[
                float(board_base[0]) + x_var,
                float(board_base[1]) + y_var,
                float(board_base[2]),
            ],  # Board position
            rotation_quat=rot_quat.tolist(),  # z-axis rotation
            name="board_with_hole",
        )
        avoid += [self.board_with_hole]

        ###
        ###
        ###
        ###
        ###
        # First generate target_number (put_in):
        # If put_in_color == 1: Randomly select a color, assign target count in range [put_in_range[0], put_in_range[1]]
        # If put_in_color == 3:
        # First generate total target count total_target from put_in_range
        # Start from [0, 0, 0], randomly distribute to three colors (no requirement for min 1 per color)

        # Then generate spawn_number:
        # If num_colors == 1: Only the color with target will spawn cube, spawn count = max(total_spawn, target count)
        # If num_colors == 3: Spawn count for each color at least equals target, remaining spawn count distributed randomly
        # This ensures spawn >= target for each color.

        # Get configuration for current difficulty
        config = self._sampling["parameters"]["configs"][self.difficulty]
        num_colors = config["color"]  # 1 or 3
        spawn_range = config["spawn_cubes"]  # [min, max]
        put_in_color_range = config["put_in_color"]
        spec_counts = None
        if spec is not None:
            order = ["red", "blue", "green"]
            spec_counts = (
                [int(spec["objects"]["spawn_count"].get(name, 0)) for name in order],
                [int(spec["objects"]["target_count"].get(name, 0)) for name in order],
            )
        color_pool = torch.randperm(3, generator=generator).tolist()[:num_colors]
        put_in_color = torch.randint(
            put_in_color_range[0], put_in_color_range[1] + 1, (1,), generator=generator
        ).item()
        put_in_color = max(1, min(3, put_in_color))
        put_in_color = min(put_in_color, max(1, num_colors))
        if spec is None:
            color_pool = self._spec.value("objects.color_pool", color_pool)
            put_in_color = self._spec.value("objects.put_in_color", put_in_color)
        active_color_indices = color_pool[:put_in_color]
        put_in_range = config["put_in_numbers"]  # [min, max]

        # First generate target_number (put_in)
        target_numbers = [0, 0, 0]
        if spec_counts is not None:
            spawn_numbers, target_numbers = list(spec_counts[0]), list(spec_counts[1])
        elif put_in_color == 1:
            # Only one color needs to be put in bin
            selected_idx = active_color_indices[0]
            target_numbers[selected_idx] = torch.randint(
                put_in_range[0], put_in_range[1] + 1, (1,), generator=generator
            ).item()
        else:
            # All 3 colors need to be put in bin, generate total number first then distribute
            total_target = torch.randint(
                put_in_range[0], put_in_range[1] + 1, (1,), generator=generator
            ).item()
            # Randomly distribute target number to three colors
            for _ in range(total_target):
                idx = torch.randint(
                    0, len(active_color_indices), (1,), generator=generator
                ).item()
                target_numbers[active_color_indices[idx]] += 1

        self.red_cubes_target_number = target_numbers[0]
        self.blue_cubes_target_number = target_numbers[1]
        self.green_cubes_target_number = target_numbers[2]

        # Then generate spawn_number, ensure spawn >= target
        if spec_counts is not None:
            total_spawn = sum(spawn_numbers)
        else:
            total_spawn = torch.randint(
                spawn_range[0], spawn_range[1] + 1, (1,), generator=generator
            ).item()

        if spec_counts is not None:
            pass
        elif num_colors == 1:
            # Only one color has cube, choose the one with target (if none, use first color in color_pool)
            spawn_numbers = [0, 0, 0]
            active_idx = next(
                (i for i in color_pool if target_numbers[i] > 0), color_pool[0]
            )
            # Spawn number at least equals target number
            spawn_numbers[active_idx] = max(total_spawn, target_numbers[active_idx])
        else:
            # num_colors controls 1/2/3 colors: ensure each selected color has at least 1 spawn, and spawn >= target
            spawn_numbers = [0, 0, 0]
            for i in color_pool:
                spawn_numbers[i] = max(target_numbers[i], 1)
            used_spawn = sum(spawn_numbers[i] for i in color_pool)
            remaining = total_spawn - used_spawn
            # Randomly distribute remaining spawn count
            for _ in range(max(0, remaining)):
                idx = torch.randint(
                    0, len(color_pool), (1,), generator=generator
                ).item()
                spawn_numbers[color_pool[idx]] += 1

        if spec is None:
            spawn_numbers = self._spec.value(
                "objects.spawn_numbers",
                spawn_numbers,
                decision_key=f"configs.{self.difficulty}.spawn_cubes",
            )
            target_numbers = self._spec.value(
                "objects.target_numbers",
                target_numbers,
                decision_key=f"configs.{self.difficulty}.put_in_numbers",
            )
        self.red_cubes_spawn_number = spawn_numbers[0]
        self.blue_cubes_spawn_number = spawn_numbers[1]
        self.green_cubes_spawn_number = spawn_numbers[2]

        logger.debug(
            f"Target numbers - Red: {self.red_cubes_target_number}, Blue: {self.blue_cubes_target_number}, Green: {self.green_cubes_target_number}"
        )
        logger.debug(
            f"Spawn numbers - Red: {self.red_cubes_spawn_number}, Blue: {self.blue_cubes_spawn_number}, Green: {self.green_cubes_spawn_number}"
        )

        ###
        ###
        ###
        ###
        ###
        self.all_cubes = []
        self.red_cubes, self.blue_cubes, self.green_cubes = [], [], []

        color_info = [
            {
                "color": (1, 0, 0, 1),
                "name": "red",
                "list": self.red_cubes,
                "spawn_num": self.red_cubes_spawn_number,
            },
            {
                "color": (0, 0, 1, 1),
                "name": "blue",
                "list": self.blue_cubes,
                "spawn_num": self.blue_cubes_spawn_number,
            },
            {
                "color": (0, 1, 0, 1),
                "name": "green",
                "list": self.green_cubes,
                "spawn_num": self.green_cubes_spawn_number,
            },
        ]

        # Generate task list for all cubes and shuffle order
        cube_tasks = []
        for info in color_info:
            for idx in range(info["spawn_num"]):
                cube_tasks.append(
                    {
                        "color": info["color"],
                        "name": info["name"],
                        "list": info["list"],
                        "idx": idx,
                    }
                )

        if spec is None:
            # Shuffle generation order
            shuffle_order = self._spec.value(
                "objects.spawn_order",
                torch.randperm(len(cube_tasks), generator=generator).tolist(),
            )
            cube_tasks = [cube_tasks[i] for i in shuffle_order]
        else:
            by_identity = {(item["name"], item["idx"]): item for item in cube_tasks}
            ordered = []
            for entry in spec["layout"]["cubes"]:
                key = (entry["color"], int(entry["color_index"]))
                if key not in by_identity:
                    raise ValueError(
                        f"spec entry {entry['object_id']} is not in the task list expanded from spawn_count"
                    )
                task = dict(by_identity[key])
                task["fixed_xy"] = [float(entry["xy"][0]), float(entry["xy"][1])]
                task["fixed_yaw"] = float(entry["yaw_rad"])
                ordered.append(task)
            if len(ordered) != len(cube_tasks):
                raise ValueError(
                    f"spec lists {len(ordered)} cubes but spawn_count expands to {len(cube_tasks)}"
                )
            cube_tasks = ordered

        # Spawn cubes in shuffled order
        cubes_cfg = self._sampling["positions"]["cubes"]
        min_gap = (
            float(cubes_cfg["min_gap_value"])
            if is_newvalue_difficulty(self.difficulty)
            else self.cube_half_size
        )
        if spec is None and is_newvalue_difficulty(self.difficulty):
            self._spawn_cubes_xhard(cube_tasks, avoid, cubes_cfg, min_gap, generator)
        else:
            for task in cube_tasks:
                try:
                    cube = spawn_random_cube(
                        self,
                        color=task["color"],
                        avoid=avoid,
                        include_existing=cubes_cfg["include_existing"],
                        include_goal=cubes_cfg["include_goal"],
                        region_center=list(cubes_cfg["region_center"]),
                        region_half_size=list(cubes_cfg["region_half_size"]),
                        half_size=self.cube_half_size,
                        min_gap=min_gap,
                        random_yaw=cubes_cfg["random_yaw"],
                        name_prefix=f"cube_{task['name']}_{task['idx']}",
                        generator=generator,
                        fixed_xy=task.get("fixed_xy"),
                        fixed_yaw=task.get("fixed_yaw"),
                        recorder=None if spec is not None else self._spec,
                        spec_path=None
                        if spec is not None
                        else f"layout.cubes.{task['name']}_{task['idx']}",
                    )
                    self.all_cubes.append(cube)
                    task["list"].append(cube)
                    avoid.append(cube)
                except RuntimeError as e:
                    logger.debug(
                        f"Failed to spawn {task['name']} cube {task['idx']}: {e}"
                    )

        logger.debug(
            f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})"
        )

        if is_newvalue_difficulty(self.difficulty):
            # Unlike the other tiers (silent shortfall), xhard fails the episode if fewer cubes than requested were placed.
            requested, actual = len(cube_tasks), len(self.all_cubes)
            self._spec.record("objects.spawn_requested", requested)
            self._spec.record("objects.spawn_actual", actual)
            if actual != requested:
                raise SceneGenerationError(
                    f"BinFill xhard: only placed {actual}/{requested} cubes"
                    f" (red {len(self.red_cubes)}/{self.red_cubes_spawn_number}, "
                    f"blue {len(self.blue_cubes)}/{self.blue_cubes_spawn_number}, "
                    f"green {len(self.green_cubes)}/{self.green_cubes_spawn_number})"
                )

        if spec is not None:
            self._injection_evidence = {
                "spec_sha256": spec.get("spec_sha256"),
                "episode": spec.get("episode"),
                "button_xy": [float(v) for v in spec["layout"]["button_xy"]],
                "board": dict(spec["layout"]["board"]),
                "spawn_count": dict(spec["objects"]["spawn_count"]),
                "target_count": dict(spec["objects"]["target_count"]),
                "cubes": [
                    {
                        "object_id": entry["object_id"],
                        "requested_xy": [float(v) for v in entry["xy"]],
                        "requested_yaw_rad": float(entry["yaw_rad"]),
                        "actual_p": _actor_xyz(cube),
                        "actual_q": _actor_quat(cube),
                    }
                    for entry, cube in zip(spec["layout"]["cubes"], self.all_cubes)
                ],
            }

    def _spawn_cubes_xhard(self, cube_tasks, avoid, cubes_cfg, min_gap, generator):
        """V5 xhard: place slots (geometry only), assign colors with V4 semantics, then build the actors.

        * **Slots**: for each slot, sample exactly as the ``spawn_random_cube`` rejection loop does (each
          trial draws u1, u2 and yaw in order; the candidate cube, padded by ``min_gap``, is tested for
          OBB intersection against the obstacles), but **do not build actors**. Placed slots enter the
          obstacle set as precomputed ``cube_obb2d_exact`` triples, so the nominal 2 cm gap takes effect.
          Accepted poses are injected via ``layout.slots.<k>``; on replay the frozen values are re-checked
          by the same rule.
        * **Colors**: slot k first takes ``cube_tasks[k]`` (the V4 ``spawn_order`` semantics). If the largest
          same-color connected cluster (center distance <= ``link_m``) exceeds ``max_component``, one
          ``randperm(n)`` reshuffle is **appended at the end**, up to ``max_redraws`` times; on failure the
          redraw with the smallest cluster wins (earliest on ties). A reshuffle only swaps color labels, not
          positions; ``recorder.value("objects.slot_assignment")`` is called only for the accepted one, and
          the number of redraws is recorded via ``record()``.
        * **Actors**: build cubes in slot order with ``spawn_random_cube(fixed_xy, fixed_yaw)`` (no random
          numbers drawn); each color list follows slot order (same as V4 "list order = spawn order");
          ``layout.cubes.*`` is now written via ``record()``.

        If a slot cannot be placed, ``SceneGenerationError`` is raised (same as V4 "a missing cube fails the episode").
        """
        half = float(self.cube_half_size)
        max_trials = 256  # must match the spawn_random_cube default
        obstacles = []
        for item in avoid:
            if (
                isinstance(item, tuple)
                and len(item) == 3
                and isinstance(item[0], np.ndarray)
            ):
                obstacles.append(item)
            elif hasattr(item, "_board_side") and hasattr(item, "_hole_side"):
                obstacles.extend(_board_strips_obb2d(item))
            else:
                raise TypeError(
                    f"BinFill xhard: unrecognized obstacle {item!r} (expected only the button OBB and the board)"
                )
        center = np.array(cubes_cfg["region_center"], dtype=np.float64)
        area_half = np.array(cubes_cfg["region_half_size"], dtype=np.float64)
        x_low, x_high = center[0] - area_half[0] + half, center[0] + area_half[0] - half
        y_low, y_high = center[1] - area_half[1] + half, center[1] + area_half[1] - half
        random_yaw = bool(cubes_cfg["random_yaw"])

        slots = []
        for k in range(len(cube_tasks)):
            placed = None
            for _trial in range(max_trials):
                u1 = torch.rand(1, generator=generator).item()
                u2 = torch.rand(1, generator=generator).item()
                x = float(x_low + u1 * (x_high - x_low))
                y = float(y_low + u2 * (y_high - y_low))
                if random_yaw:
                    yaw = float(torch.rand(1, generator=generator).item() * 2 * np.pi)
                else:
                    yaw = 0.0
                c_new, A_new, h_new = _build_new_cube_obb2d(
                    x, y, half, yaw, pad_xy=float(min_gap)
                )
                if any(
                    _obb2d_intersect(c, A, h, c_new, A_new, h_new)
                    for (c, A, h) in obstacles
                ):
                    continue
                placed = [x, y, yaw]
                break
            if placed is None:
                raise SceneGenerationError(
                    f"BinFill xhard: slot {k} could not be placed within {max_trials} attempts ({k}/{len(cube_tasks)} placed)"
                )
            x, y, yaw = (
                float(v) for v in self._spec.value(f"layout.slots.{k}", placed)
            )
            if not (x_low <= x <= x_high and y_low <= y <= y_high):
                raise EpisodeSpecError(
                    f"BinFill xhard: slot {k} center ({x}, {y}) is outside the cube region"
                )
            c_new, A_new, h_new = _build_new_cube_obb2d(
                x, y, half, yaw, pad_xy=float(min_gap)
            )
            if any(
                _obb2d_intersect(c, A, h, c_new, A_new, h_new)
                for (c, A, h) in obstacles
            ):
                raise EpisodeSpecError(
                    f"BinFill xhard: frozen pose ({x}, {y}, {yaw}) of slot {k} is closer than {min_gap} to the button, board or placed cubes"
                )
            obstacles.append(cube_obb2d_exact((x, y, yaw), half))
            slots.append((x, y, yaw))

        mix = self._sampling["parameters"]["configs"][self.difficulty]["color_mix"]
        max_component = int(mix["max_component"])
        link = float(mix["link_m"])
        max_redraws = int(mix["max_redraws"])
        n = len(cube_tasks)
        xy = np.array([[s[0], s[1]] for s in slots], dtype=np.float64).reshape(-1, 2)
        names = [task["name"] for task in cube_tasks]

        def _component(order):
            return _max_same_color_component(xy, [names[i] for i in order], link)

        best_order = list(range(n))
        best_component = _component(best_order)
        redraws = 0
        while best_component > max_component and redraws < max_redraws:
            perm = torch.randperm(n, generator=generator).tolist()
            redraws += 1
            component = _component(perm)
            if component < best_component:
                best_order, best_component = perm, component
        fallback = best_component > max_component
        self._spec.record("objects.color_redraws", redraws)
        self._spec.record("objects.color_mix_fallback", bool(fallback))
        self._spec.record("objects.color_mix_max_component", int(best_component))
        by_identity = {f"{task['name']}_{task['idx']}": task for task in cube_tasks}
        assignment = [
            f"{cube_tasks[i]['name']}_{cube_tasks[i]['idx']}" for i in best_order
        ]
        assignment = list(self._spec.value("objects.slot_assignment", assignment))
        if sorted(assignment) != sorted(by_identity):
            raise EpisodeSpecError(
                f"BinFill xhard: frozen slot_assignment {assignment} is not the same set as this episode's cube identities {sorted(by_identity)}"
            )
        frozen_component = _max_same_color_component(
            xy, [by_identity[ident]["name"] for ident in assignment], link
        )
        if frozen_component > max(max_component, best_component):
            raise EpisodeSpecError(
                f"BinFill xhard: frozen slot_assignment has a largest same-color cluster of {frozen_component}, above the limit {max_component}"
            )

        for k, ident in enumerate(assignment):
            task = by_identity[ident]
            x, y, yaw = slots[k]
            cube = spawn_random_cube(
                self,
                color=task["color"],
                avoid=None,
                include_existing=False,
                include_goal=False,
                region_center=list(cubes_cfg["region_center"]),
                region_half_size=list(cubes_cfg["region_half_size"]),
                half_size=self.cube_half_size,
                min_gap=min_gap,
                random_yaw=cubes_cfg["random_yaw"],
                name_prefix=f"cube_{task['name']}_{task['idx']}",
                generator=None,
                fixed_xy=[x, y],
                fixed_yaw=yaw,
            )
            self._spec.record(f"layout.cubes.{ident}", [x, y, yaw])
            self.all_cubes.append(cube)
            task["list"].append(cube)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._native_init_index = getattr(self, "_native_init_index", -1) + 1
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos = reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)

            tasks = []
            self.red_cubes_in_bin = 0
            self.blue_cubes_in_bin = 0
            self.green_cubes_in_bin = 0
            self.binfill_language_sequence = []
            color_task_definitions = [
                ("blue", self.blue_cubes, self.blue_cubes_target_number),
                ("red", self.red_cubes, self.red_cubes_target_number),
                ("green", self.green_cubes, self.green_cubes_target_number),
            ]
            color_order = self._spec.value(
                f"initializations.{self._native_init_index}.color_order",
                torch.randperm(
                    len(color_task_definitions), generator=self.generator
                ).tolist(),
            )
            for color_idx in color_order:
                color_name, cube_collection, target_number = color_task_definitions[
                    color_idx
                ]
                if target_number <= 0:
                    continue
                if (
                    is_newvalue_difficulty(self.difficulty)
                    and len(cube_collection) < target_number
                ):
                    raise SceneGenerationError(
                        f"BinFill xhard: {color_name} has only {len(cube_collection)} cubes but {target_number} must be put in"
                    )
                self.binfill_language_sequence.append((color_name, target_number))
                for i in range(target_number):
                    cube = cube_collection[i]
                    tasks.append(
                        {
                            "func": lambda c=self.all_cubes: (
                                is_any_obj_pickup_flag_currentpickup(self, objects=c)
                            ),
                            "name": subgoal_language.get_subgoal_with_index(
                                i, "pick up the {idx} {color} cube", color=color_name
                            ),
                            "subgoal_segment": subgoal_language.get_subgoal_with_index(
                                i,
                                "pick up the {idx} {color} cube at <>",
                                color=color_name,
                            ),
                            "choice_label": "pick up the cube",
                            "demonstration": False,
                            "failure_reason": "premature_button",
                            "failure_func": lambda: is_button_pressed(
                                self, obj=self.button
                            ),
                            "solve": lambda env, planner, c=cube: solve_pickup(
                                env, planner, obj=c
                            ),
                            "segment": [cube_collection[i]],
                        }
                    )
                    tasks.append(
                        {
                            "func": lambda c=self.all_cubes: (
                                is_any_obj_dropped_onto_delete(
                                    self, objects=c, target=self.board_with_hole
                                )
                            ),
                            "name": f"put it into the bin",
                            "subgoal_segment": "put it into the bin at <>",
                            "choice_label": "put it into the bin",
                            "demonstration": False,
                            "failure_reason": "premature_button",
                            "failure_func": lambda: is_button_pressed(
                                self, obj=self.button
                            ),
                            "solve": lambda env, planner, c=cube: [
                                solve_putonto_whenhold_binspecial(
                                    env, planner, target=self.board_with_hole
                                ),
                            ],
                            "segment": [self.board_with_hole],
                        }
                    )
            tasks.append(
                {
                    "func": lambda: is_button_pressed(self, obj=self.button),
                    "name": "press the button",
                    "subgoal_segment": "press the button at <>",
                    "choice_label": "press the button",
                    "demonstration": False,
                    "failure_reason": ["incorrect_bin_count", "extra_cube_in_bin"],
                    "failure_func": lambda c=self.all_cubes: [
                        not check_in_bin_number(
                            self,
                            in_bin_list=[
                                self.red_cubes_in_bin,
                                self.blue_cubes_in_bin,
                                self.green_cubes_in_bin,
                            ],
                            total_number_list=[
                                self.red_cubes_target_number,
                                self.blue_cubes_target_number,
                                self.green_cubes_target_number,
                            ],
                        ),
                        is_any_obj_dropped_onto_delete(
                            self, objects=c, target=self.board_with_hole
                        ),
                    ],
                    "solve": lambda env, planner: [
                        solve_button(env, planner, obj=self.button)
                    ],
                    "segment": self.cap_link,
                }
            )
            self.task_list = tasks
            # Record pickup related task indices and items for recovery
            self.fail_grasp_task_index = None

    def evaluate(self, solve_complete_eval=False):
        self.successflag = torch.tensor([False])
        # Save current_task_failure state before calling sequential_task_check
        # This is because failure might be detected during step(), but sequential_task_check might reset it
        previous_failure = getattr(self, "current_task_failure", False)
        self.failureflag = torch.tensor([False])

        if (
            self.use_demonstrationwrapper == False
        ):  # change subgoal after planner ends during recording
            if solve_complete_eval == True:
                allow_subgoal_change_this_timestep = True
            else:
                allow_subgoal_change_this_timestep = False
        else:  # during demonstration, video needs to call evaluate(solve_complete_eval), video ends and flag changes in demonstrationwrapper
            if solve_complete_eval == True or self.demonstration_record_traj == False:
                allow_subgoal_change_this_timestep = True
            else:
                allow_subgoal_change_this_timestep = False

        # Use encapsulated sequence task check function
        (
            all_tasks_completed,
            current_task_name,
            task_failed,
            self.current_task_specialflag,
        ) = sequential_task_check(
            self,
            self.task_list,
            allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep,
        )

        # If task failed, mark as failed immediately
        # Or if failure was detected previously (previous_failure), also mark as failed
        if task_failed or previous_failure:
            self.failureflag = torch.tensor([True])
            if task_failed:
                logger.debug(f"Task failed: {current_task_name}")
            elif previous_failure:
                # If marked failed due to previous_failure, ensure current_task_failure is also set
                self.current_task_failure = True

        # If static_check succeeds or all tasks completed, set success flag
        if all_tasks_completed and not task_failed:
            self.successflag = torch.tensor([True])

        return {
            "success": self.successflag,
            "fail": self.failureflag,
        }

    # Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        self.vis_obj_id_list = []

        timestep = self.elapsed_steps
        if self.dynamic:
            # Dynamically lift cubes for each color (starting from 2nd cube)
            for cube_list in [self.red_cubes, self.blue_cubes, self.green_cubes]:
                for idx in range(1, len(cube_list)):
                    lift_and_drop_objects_back_to_original(
                        self,
                        obj=cube_list[idx],
                        start_step=0,
                        end_step=idx * 100,
                        cur_step=timestep,
                    )

        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info


from robomme_plus.registry import TASKS
from robomme_plus.tasks.counting.utils.environment import CountingMixin


@register_env(TASKS["BinFill"].gym_id)
class BinFill(CountingMixin, _TaskScene):
    """Fill the bin with requested cubes from a moving conveyor, then press the button."""
    PERCEPTUAL_VARIATIONS = True

    task_name = "BinFill"
