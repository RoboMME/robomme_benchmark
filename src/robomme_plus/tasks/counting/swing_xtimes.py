"""Swing Xtimes: scene construction, native scoring and RoboMME-Plus task configuration."""

import copy
from typing import Dict, Union

import numpy as np
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

import robomme_plus.control.reset as reset_panda
import robomme_plus.scoring.language as subgoal_language
from robomme_plus.simulation.base import MemoryTask
from robomme_plus.simulation.difficulty import (
    is_newvalue_difficulty,
    normalize_robomme_difficulty,
)
from robomme_plus.simulation.specification import SpecRecorder

# Robomme
from robomme_plus.common.errors import SceneGenerationError
from robomme_plus.common.errors import (
    SceneGenerationError as _RealSceneGenerationError,
)
from robomme_plus.common.logging_utils import logger
from robomme_plus.simulation.actors import (
    build_button,
    spawn_random_cube,
    spawn_random_target,
)
from robomme_plus.control.planning import (
    solve_button,
    solve_pickup,
    solve_putdown_whenhold,
    solve_swingonto_whenhold,
)
from robomme_plus.scoring.predicates import (
    is_any_obj_pickup,
    is_button_pressed,
    is_obj_dropped,
    is_obj_pickup,
    is_obj_swing_onto,
    sequential_task_check,
    too_many_swings,
)
from robomme_plus.simulation.choreography import highlight_obj
from robomme_plus.simulation.geometry import (
    BLOCK_DISTRACTOR_COLORS,
    cube_obb2d_exact,
)


SCENE_CONFIG = {'parameters': {'cubes_per_color': 1,
                'color_pool': [{'rgba': [1, 0, 0, 1], 'name': 'red'},
                               {'rgba': [0, 0, 1, 1], 'name': 'blue'},
                               {'rgba': [0, 1, 0, 1], 'name': 'green'}],
                'color_and_target_selection': {'shuffle': 'torch.randperm(len(color_groups))',
                                               'target_color_idx': 'torch.randint(0, '
                                                                   'len(color_groups), (1,))',
                                               'target_cube_idx': 'torch.randint(0, '
                                                                  'len(all_cubes), (1,))'},
                'side_order': {'first': 'right', 'second': 'left', 'max_swings': '2 * num_repeats'},
                'swing_thresholds': {'distance': 0.03, 'z': 0.12, 'height': 0.1},
                'recovery': 'reuses the fail recover mode given at the entry point and the '
                            'original generator'},
 'positions': {'button': {'center_xy': [-0.2, 0], 'scale': 1.5},
               'cubes': {'region_center': [-0.1, 0],
                         'region_half_size': 0.25,
                         'random_yaw': True,
                         'min_gap': 'self.cube_half_size'},
               'targets': [{'region_center': [-0.1, -0.2],
                            'region_half_size': 0.1,
                            'name': 'temp_target_0'},
                           {'region_center': [-0.1, 0.2],
                            'region_half_size': 0.1,
                            'name': 'temp_target_1'}],
               'target_geometry': {'radius_factor': 2,
                                   'thickness': 0.005,
                                   'min_gap_factor': 1,
                                   'style': 'gray'}},
 'decision': {'number_range': {'hard': [3, 3], 'xhard1': [4, 4]},
              'color': {'hard': 3, 'xhard1': 3},
              'distractor': None,
              'xhard1': {'distractor': {'colors': ['yellow'],
                                        'region_center': [-0.1, 0],
                                        'region_half_size': 0.25},
                         'min_center_dist_m': 0.08}}}


def _scene_gen_error(difficulty):
    """Select the scene-generation exception class by difficulty.

    xhard-family difficulties return the real ``SceneGenerationError`` (a retryable task failure); the original
    three difficulties return the shadowed module-level name ``SceneGenerationError`` (the submodule, which is
    still a TypeError on raise / except, behavior unchanged).
    Usage: ``raise _scene_gen_error(self.difficulty)("message")``, ``except _scene_gen_error(self.difficulty):``;
    code that only runs on the xhard path uses ``_RealSceneGenerationError`` directly.
    """
    return (
        _RealSceneGenerationError
        if is_newvalue_difficulty(difficulty)
        else SceneGenerationError
    )


# Distractor colors must stay out of native.color_pool: merging changes randperm(len(color_groups)) and shifts the original random stream.


def _disk_avoid_obb(target, clearance):
    """Convert a disk into a precomputed OBB ``(center, axes, half extents)`` usable by cube rejection sampling.

    The disk is a visual-only actor with ``add_collision=False``, so ``get_actor_obb`` cannot get a mesh, and
    putting it directly into ``avoid`` is silently ignored by ``spawn_random_cube`` (observed 2026-09-22).
    Distractor cubes are placed after the disks, so a bounding square must be given explicitly:
    half extent = disk radius + disk gap - the cube's own gap.
    """
    p = target.pose.p
    if isinstance(p, torch.Tensor):
        p = p[0].detach().cpu().numpy()
    return (
        np.array(p[:2], dtype=np.float64),
        np.eye(2, dtype=np.float64),
        np.array([clearance, clearance], dtype=np.float64),
    )


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
        native_episode_spec=None,
        **kwargs,
    ):
        # Must run before any random-number call and before super().__init__()
        self._sampling = copy.deepcopy(SCENE_CONFIG)
        self._spec = SpecRecorder(
            native_episode_spec,
            "SwingXtimes",
            {"seed": seed},
            difficulty=kwargs.get("difficulty"),
        )
        self._native_init_index = -1
        self.use_demonstrationwrapper = False
        self.demonstration_record_traj = False
        self.robot_init_qpos_noise = robot_init_qpos_noise
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
        normalized_robomme_difficulty = normalize_robomme_difficulty(
            kwargs.pop("difficulty", None)
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

            # Use seed to randomly determine number of repetitions (1-5)
        generator = torch.Generator()
        generator.manual_seed(seed)
        number_range = self._sampling["decision"]["number_range"][self.difficulty]
        self.num_repeats = self._spec.value(
            "objects.num_repeats",
            torch.randint(
                number_range[0], number_range[1] + 1, (1,), generator=generator
            ).item(),
            decision_key=f"number_range.{self.difficulty}",
        )
        self._spec.identity.setdefault("difficulty", self.difficulty)
        logger.debug(f"Task will repeat {self.num_repeats} times (pickup-drop cycles)")

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        try:
            self.table_scene = TableSceneBuilder(
                self, robot_init_qpos_noise=self.robot_init_qpos_noise
            )
            self.table_scene.build()

            button_cfg = self._sampling["positions"]["button"]
            cubes_cfg = self._sampling["positions"]["cubes"]
            targets_cfg = self._sampling["positions"]["targets"]
            target_geom = self._sampling["positions"]["target_geometry"]
            button_obb = build_button(
                self,
                center_xy=tuple(button_cfg["center_xy"]),
                scale=button_cfg["scale"],
                generator=generator,
                recorder=self._spec,
                spec_path="layout.button_xy",
            )
            avoid = [button_obb]

            self.all_cubes = []  # Save all cube objects

            # Initialize storage for each color group
            self.red_cubes = []
            self.red_cube_names = []
            self.blue_cubes = []
            self.blue_cube_names = []
            self.green_cubes = []
            self.green_cube_names = []

            cubes_per_color = self._sampling["parameters"]["cubes_per_color"]
            _color_lists = {
                "red": (self.red_cubes, self.red_cube_names),
                "blue": (self.blue_cubes, self.blue_cube_names),
                "green": (self.green_cubes, self.green_cube_names),
            }
            for entry in self._sampling["parameters"]["color_pool"]:
                if entry["name"] not in _color_lists:
                    extra_cubes, extra_names = [], []
                    setattr(self, f"{entry['name']}_cubes", extra_cubes)
                    setattr(self, f"{entry['name']}_cube_names", extra_names)
                    _color_lists[entry["name"]] = (extra_cubes, extra_names)
            self._cube_color_of = []
            color_groups = [
                {
                    "color": tuple(entry["rgba"]),
                    "name": entry["name"],
                    "list": _color_lists[entry["name"]][0],
                    "name_list": _color_lists[entry["name"]][1],
                }
                for entry in self._sampling["parameters"]["color_pool"]
            ]
            shuffle_indices = self._spec.value(
                "objects.color_order",
                torch.randperm(len(color_groups), generator=generator).tolist(),
            )
            color_groups = [color_groups[i] for i in shuffle_indices]

            # Randomly select target color using generator
            target_color_idx = self._spec.value(
                "objects.target_color_idx",
                torch.randint(0, len(color_groups), (1,), generator=generator).item(),
            )
            self.target_color_name = color_groups[target_color_idx]["name"]
            logger.debug(f"Target color selected: {self.target_color_name}")

            if is_newvalue_difficulty(self.difficulty):
                self._spawn_colored_cubes_xhard(generator, avoid, color_groups)
            else:
                # Generate cubes for each color group
                for idx, group in enumerate(color_groups):
                    if idx < self._sampling["decision"]["color"][self.difficulty]:
                        for cube_idx in range(cubes_per_color):
                            try:
                                cube = spawn_random_cube(
                                    self,
                                    color=group["color"],
                                    avoid=avoid,
                                    include_existing=False,
                                    include_goal=False,
                                    region_center=list(cubes_cfg["region_center"]),
                                    region_half_size=cubes_cfg["region_half_size"],
                                    half_size=self.cube_half_size,
                                    min_gap=self.cube_half_size,
                                    random_yaw=cubes_cfg["random_yaw"],
                                    name_prefix=f"cube_{group['name']}_{cube_idx}",
                                    generator=generator,
                                    recorder=self._spec,
                                    spec_path=f"layout.cubes.{group['name']}_{cube_idx}",
                                )
                            except RuntimeError as e:
                                raise _scene_gen_error(self.difficulty)(
                                    f"Failed to generate {group['name']} cube {cube_idx}: {e}"
                                ) from e

                            self.all_cubes.append(cube)
                            group["list"].append(cube)
                            cube_name = f"cube_{group['name']}_{cube_idx}"
                            group["name_list"].append(cube_name)
                            self._cube_color_of.append((cube, group["name"]))
                            setattr(self, cube_name, cube)
                            avoid.append(cube)

                    logger.debug(
                        f"Generated {len(group['list'])} {group['name']} cubes"
                    )

            logger.debug(
                f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})"
            )

            # Generate first target
            try:
                temp_target_0 = spawn_random_target(
                    self,
                    avoid=avoid,  # Use current avoidance list, containing all spawned cubes
                    include_existing=False,  # Manually maintain list
                    include_goal=False,  # Manually maintain list
                    region_center=list(targets_cfg[0]["region_center"]),
                    region_half_size=targets_cfg[0]["region_half_size"],
                    radius=self.cube_half_size
                    * target_geom["radius_factor"],  # Use radius instead of half_size
                    thickness=target_geom["thickness"],  # target thickness
                    min_gap=self.cube_half_size
                    * target_geom["min_gap_factor"],  # Gap requirement same as cube
                    name_prefix=f"temp_target_0",
                    generator=generator,
                    target_style="gray",
                    recorder=self._spec,
                    spec_path="layout.targets.0",
                )
                avoid.append(temp_target_0)
                logger.debug(f"Generated first target")
            except RuntimeError as e:
                raise _scene_gen_error(self.difficulty)(
                    "First target sampling failed"
                ) from e

            # Generate second target
            try:
                temp_target_1 = spawn_random_target(
                    self,
                    avoid=avoid,  # Use current avoidance list, containing all spawned cubes and first target
                    include_existing=False,  # Manually maintain list
                    include_goal=False,  # Manually maintain list
                    region_center=list(targets_cfg[1]["region_center"]),
                    region_half_size=targets_cfg[1]["region_half_size"],
                    radius=self.cube_half_size
                    * target_geom["radius_factor"],  # Use radius instead of half_size
                    thickness=target_geom["thickness"],  # target thickness
                    min_gap=self.cube_half_size
                    * target_geom["min_gap_factor"],  # Gap requirement same as cube
                    name_prefix=f"temp_target_1",
                    generator=generator,
                    target_style="gray",
                    recorder=self._spec,
                    spec_path="layout.targets.1",
                )
                avoid.append(temp_target_1)
                logger.debug(f"Generated second target")
            except RuntimeError as e:
                raise _scene_gen_error(self.difficulty)(
                    "Second target sampling failed"
                ) from e

            # Swap names if necessary to ensure target_0.y < target_1.y
            temp_0_y = temp_target_0.pose.p[0, 1].item()  # Get y coordinate
            temp_1_y = temp_target_1.pose.p[0, 1].item()  # Get y coordinate

            if temp_0_y < temp_1_y:
                # No swap needed
                self.target_right = temp_target_0
                self.target_left = temp_target_1
                logger.debug(
                    f"target_0 y={temp_0_y:.3f}, target_1 y={temp_1_y:.3f} (no swap needed)"
                )
            else:
                # Swap the assignments
                self.target_right = temp_target_1
                self.target_left = temp_target_0
                logger.debug(
                    f"Swapped: target_0 y={temp_1_y:.3f}, target_1 y={temp_0_y:.3f} (swapped to ensure target_0.y < target_1.y)"
                )

            if is_newvalue_difficulty(self.difficulty):
                self._select_target_xhard(generator)
            # Randomly select one cube from all available cubes as the target
            elif len(self.all_cubes) > 0:
                target_cube_idx = self._spec.value(
                    "objects.target_cube_idx",
                    torch.randint(
                        0, len(self.all_cubes), (1,), generator=generator
                    ).item(),
                )
                self.target_cube = self.all_cubes[target_cube_idx]

                # Determine the color of the selected target cube
                if self.target_cube in self.red_cubes:
                    self.target_color_name = "red"
                elif self.target_cube in self.blue_cubes:
                    self.target_color_name = "blue"
                elif self.target_cube in self.green_cubes:
                    self.target_color_name = "green"

                logger.debug(
                    f"Target cube selected: {self.target_color_name} cube (index {target_cube_idx} in all_cubes)"
                )
            else:
                self.target_cube = None
                self.target_color_name = None
                logger.debug("No cubes generated, no target cube selected")

            # Create list of non-target cubes for failure checking
            self.non_target_cubes = [
                cube for cube in self.all_cubes if cube != self.target_cube
            ]
            logger.debug(f"Non-target cubes: {len(self.non_target_cubes)}")

        except _scene_gen_error(self.difficulty):
            raise
        except Exception as exc:
            raise _scene_gen_error(self.difficulty)(
                f"Failed to load SwingXtimes scene for seed {self.seed}"
            ) from exc

        tasks = []
        tasks.append(
            {
                "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                "name": f"pick up the {self.target_color_name} cube",
                "subgoal_segment": f"pick up the {self.target_color_name} cube at <>",
                "choice_label": "pick up the cube",
                "demonstration": False,
                "failure_reason": ["wrong_cube", "premature_button", "too_many_swings"],
                "failure_func": lambda: [
                    is_any_obj_pickup(self, self.non_target_cubes),
                    is_button_pressed(self, obj=self.button),
                    too_many_swings(self),
                ],
                "solve": lambda env, planner: solve_pickup(
                    env, planner, obj=self.target_cube
                ),
                "segment": self.target_cube,
            }
        )

        _swing_cfg = self._sampling["parameters"]["swing_thresholds"]
        sides = ("right", "left") if self.swing_start_side == "right" else ("left", "right")
        for i in range(self.num_repeats):
            ordinal = subgoal_language._ordinal_word(i)
            for side in sides:
                target = getattr(self, f"target_{side}")
                tasks.append(
                    {
                        "func": lambda target=target: is_obj_swing_onto(
                            self,
                            obj=self.target_cube,
                            target=target,
                            distance_threshold=_swing_cfg["distance"],
                            z_threshold=_swing_cfg["z"],
                        ),
                        "name": f"move to the top of the {side}-side target for the {ordinal} time",
                        "subgoal_segment": f"move to the top of the {side}-side target at <> for the {ordinal} time",
                        "choice_label": "move to the top of the target",
                        "demonstration": False,
                        "failure_reason": ["wrong_cube", "premature_button", "too_many_swings"],
                        "failure_func": lambda: [
                            is_any_obj_pickup(self, self.non_target_cubes),
                            is_button_pressed(self, obj=self.button),
                            too_many_swings(self),
                        ],
                        "solve": lambda env, planner, target=target: [
                            solve_swingonto_whenhold(
                                env, planner, target=target,
                                height=_swing_cfg["height"],
                            ),
                        ],
                        "segment": target,
                    }
                )

        tasks.append(
            {
                "func": (lambda: is_obj_dropped(self, obj=self.target_cube)),
                "name": f"put the {self.target_color_name} cube on the table",
                "subgoal_segment": f"put the {self.target_color_name} cube on the table",
                "choice_label": "put the cube on the table",
                "demonstration": False,
                "failure_reason": ["wrong_cube", "premature_button", "too_many_swings"],
                "failure_func": lambda: [
                    is_any_obj_pickup(self, self.non_target_cubes),
                    is_button_pressed(self, obj=self.button),
                    too_many_swings(self),
                ],
                "solve": lambda env, planner: solve_putdown_whenhold(
                    env,
                    planner,
                ),
            }
        )
        tasks.append(
            {
                "func": lambda: is_button_pressed(self, obj=self.button),
                "name": "press the button",
                "subgoal_segment": "press the button at <>",
                "choice_label": "press the button",
                "demonstration": False,
                "failure_reason": ["wrong_cube", "too_many_swings"],
                "failure_func": lambda: [
                    is_any_obj_pickup(self, self.non_target_cubes),
                    too_many_swings(self),
                ],
                "solve": lambda env, planner: solve_button(
                    env, planner, obj=self.button
                ),
                "segment": self.cap_link,
            }
        )

        # Store task list for RecordWrapper use
        self.task_list = tasks

        # Record pickup related task indices and items for recovery
        self.fail_grasp_task_index = None

        # Must come after every existing random draw (including recovery sampling) so the original random stream is unchanged (N5).
        if is_newvalue_difficulty(self.difficulty):
            self._spawn_distractors_xhard(generator, avoid)

    def _color_name_of(self, cube):
        """Look up the color name by object (xhard path); raises instead of leaving a stale value if the cube was not registered."""
        for actor, name in self._cube_color_of:
            if actor is cube:
                return name
        raise _RealSceneGenerationError(
            "SwingXtimes xhard: target cube is not in the color registry"
        )

    def _select_target_xhard(self, generator):
        """Select the target cube for xhard: draw from an explicit candidate list, with color filled in per object.

        The draw position matches ``objects.target_cube_idx`` of the original path (one randint), only the upper
        bound is the candidate pool size; the pool holds only colored cubes, so distractor cubes added later
        are never drawn as the target.
        """
        self._spec.record(
            "objects.cube_count",
            {
                "requested": min(
                    self._sampling["decision"]["color"][self.difficulty],
                    len(self._sampling["parameters"]["color_pool"]),
                )
                * self._sampling["parameters"]["cubes_per_color"],
                "actual": len(self.all_cubes),
            },
        )
        self.target_candidates = list(self.all_cubes)
        if not self.target_candidates:
            raise _RealSceneGenerationError(
                "SwingXtimes xhard: no target candidate cubes available"
            )
        self._spec.record(
            "objects.target_candidates",
            [self._color_name_of(cube) for cube in self.target_candidates],
        )
        target_cube_idx = self._spec.value(
            "objects.target_cube_idx",
            torch.randint(
                0, len(self.target_candidates), (1,), generator=generator
            ).item(),
        )
        self.target_cube = self.target_candidates[target_cube_idx]
        self.target_color_name = self._color_name_of(self.target_cube)
        self.distractor_cubes = []
        self.non_target_cubes = [
            cube for cube in self.all_cubes if cube is not self.target_cube
        ]

    def _append_cube_obstacle_xhard(self, cube, avoid):
        """Register a just-placed cube as an exact OBB obstacle and center-distance reference for later objects.

        The actor itself is no longer put into ``avoid``: the actor path goes through ``_trimesh_box_to_obb2d``,
        which degenerates to a line segment for about 2/3 of cube orientations, so ``min_gap`` fails along its
        normal. Pure geometry; draws no random numbers.
        """
        obb = cube_obb2d_exact(cube, self.cube_half_size)
        self._xhard_cube_obbs.append(obb)
        avoid.append(obb)

    def _spawn_colored_cubes_xhard(self, generator, avoid, color_groups):
        """Place the three colored cubes (target candidates) for xhard.

        Differs from the shared loop of the original difficulties in only two ways: the candidate center must be at
        least ``min_center_dist_m`` from every placed cube (via ``spawn_random_cube(min_center_dist=...)``, which
        draws no random numbers itself); and each placed cube enters ``avoid`` as an exact ``cube_obb2d_exact`` OBB
        (the two disks and the distractor cubes that follow avoid it). Region, gap, yaw, draw-point path and
        per-cube rejection budget (default 256) match the original loop; a cube that cannot be placed raises the
        real ``SceneGenerationError``.
        """
        cubes_cfg = self._sampling["positions"]["cubes"]
        cubes_per_color = self._sampling["parameters"]["cubes_per_color"]
        min_center_dist = float(
            self._sampling["decision"][self.difficulty]["min_center_dist_m"]
        )
        self._spec.record("layout.cube_min_center_dist", min_center_dist)
        self._xhard_cube_obbs = []
        for idx, group in enumerate(color_groups):
            if idx < self._sampling["decision"]["color"][self.difficulty]:
                for cube_idx in range(cubes_per_color):
                    cube_name = f"cube_{group['name']}_{cube_idx}"
                    try:
                        cube = spawn_random_cube(
                            self,
                            color=group["color"],
                            avoid=avoid,
                            include_existing=False,
                            include_goal=False,
                            region_center=list(cubes_cfg["region_center"]),
                            region_half_size=cubes_cfg["region_half_size"],
                            half_size=self.cube_half_size,
                            min_gap=self.cube_half_size,
                            random_yaw=cubes_cfg["random_yaw"],
                            name_prefix=cube_name,
                            generator=generator,
                            recorder=self._spec,
                            spec_path=f"layout.cubes.{group['name']}_{cube_idx}",
                            min_center_dist=(min_center_dist, self._xhard_cube_obbs),
                        )
                    except RuntimeError as exc:
                        raise _RealSceneGenerationError(
                            f"SwingXtimes xhard: cube {cube_name} could not be placed: {exc}"
                        ) from exc

                    self.all_cubes.append(cube)
                    group["list"].append(cube)
                    group["name_list"].append(cube_name)
                    self._cube_color_of.append((cube, group["name"]))
                    setattr(self, cube_name, cube)
                    self._append_cube_obstacle_xhard(cube, avoid)

                logger.debug(f"Generated {len(group['list'])} {group['name']} cubes")

    def _spawn_distractors_xhard(self, generator, avoid):
        """Place the distractor cubes of other colors for xhard.

        Distractor cubes go into ``all_cubes`` and ``non_target_cubes`` (grasping the wrong one triggers
        failure_func) but not into ``target_candidates``; the two disks are avoided explicitly via
        ``_disk_avoid_obb``. A cube that cannot be placed raises ``SceneGenerationError`` directly (no silent
        truncation). Shares the center-distance rule and exact OBB obstacles with the colored cubes.
        """
        min_center_dist = float(
            self._sampling["decision"][self.difficulty]["min_center_dist_m"]
        )
        dcfg = self._sampling["decision"][self.difficulty]["distractor"]
        palette = {entry["name"]: entry["rgba"] for entry in BLOCK_DISTRACTOR_COLORS}
        names = list(dcfg["colors"])
        unknown = [name for name in names if name not in palette]
        if unknown:
            raise _RealSceneGenerationError(
                f"SwingXtimes xhard: distractor color not in BLOCK_DISTRACTOR_COLORS: {unknown}"
            )
        target_geom = self._sampling["positions"]["target_geometry"]
        clearance = (
            self.cube_half_size
            * (target_geom["radius_factor"] + target_geom["min_gap_factor"])
            - self.cube_half_size
        )
        avoid = list(avoid) + [
            _disk_avoid_obb(disk, clearance)
            for disk in (self.target_right, self.target_left)
        ]
        self._spec.record(
            "objects.distractors", [{"name": f"cube_{n}_0", "color": n} for n in names]
        )
        for name in names:
            cube_name = f"cube_{name}_0"
            try:
                cube = spawn_random_cube(
                    self,
                    color=tuple(palette[name]),
                    avoid=avoid,
                    include_existing=False,
                    include_goal=False,
                    region_center=list(dcfg["region_center"]),
                    region_half_size=dcfg["region_half_size"],
                    half_size=self.cube_half_size,
                    min_gap=self.cube_half_size,
                    random_yaw=self._sampling["positions"]["cubes"]["random_yaw"],
                    name_prefix=cube_name,
                    generator=generator,
                    recorder=self._spec,
                    spec_path=f"layout.distractors.{name}_0",
                    min_center_dist=(min_center_dist, self._xhard_cube_obbs),
                )
            except RuntimeError as exc:
                raise _RealSceneGenerationError(
                    f"SwingXtimes xhard: distractor cube {cube_name} could not be placed: {exc}"
                ) from exc
            self.all_cubes.append(cube)
            self.distractor_cubes.append(cube)
            self._cube_color_of.append((cube, name))
            setattr(self, f"{name}_cubes", [cube])
            setattr(self, f"{name}_cube_names", [cube_name])
            setattr(self, cube_name, cube)
            self._append_cube_obstacle_xhard(cube, avoid)
        self._spec.record(
            "objects.distractor_count",
            {"requested": len(names), "actual": len(self.distractor_cubes)},
        )
        if len(self.distractor_cubes) != len(names):
            raise _RealSceneGenerationError(
                f"SwingXtimes xhard: requested {len(names)} distractor cubes, got {len(self.distractor_cubes)}"
            )
        # failure_func reads self.non_target_cubes at call time; rebuilding it here makes distractors count toward failure.
        self.non_target_cubes = [
            cube for cube in self.all_cubes if cube is not self.target_cube
        ]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._native_init_index = getattr(self, "_native_init_index", -1) + 1
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos = reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)
            self.highlight_right_start = None
            self.highlight_left_start = None
            # Swing count initialization:
            # swing_count     Record cumulative swing/landing counts (left and right each count as one)
            # swing_over_limit Mark whether allowed count is exceeded, if True then failed
            # _was_on_right/_was_on_left Used for "edge detection" to prevent duplicate counting of same landing across multiple frames
            self.swing_count = 0
            self.swing_over_limit = False
            self._was_on_right = False
            self._was_on_left = False
            # Expected max swing count (left and right each self.num_repeats times)
            self.max_swings = self.num_repeats * 2

    def evaluate(self, solve_complete_eval=False):
        previous_failure = getattr(self, "failureflag", None)
        self.successflag = torch.tensor([False])
        if previous_failure is not None and bool(previous_failure.item()):
            self.failureflag = previous_failure
        else:
            self.failureflag = torch.tensor([False])

        # To test "exceed swing limit" scenario, forcibly lower limit (e.g. 1 time)
        # This way second landing triggers too_many_swings failure logic, for easy verification
        # self.max_swings = 2

        # Use encapsulated sequence task check function
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
        if task_failed:
            self.failureflag = torch.tensor([True])
            logger.debug(f"Task failed: {current_task_name}")

        # If static_check succeeds or all tasks completed, set success flag
        if all_tasks_completed and not task_failed:
            self.successflag = torch.tensor([True])

        return {
            "success": self.successflag,
            "fail": self.failureflag,
        }

    # Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):

        obs, reward, terminated, truncated, info = super().step(action)
        # First check if current frame "lands" on left/right targets, for highlight and counting
        # Note: When policy jitters in z-axis, is_obj_swing_onto's z_threshold check might cause on/off flipping,
        # triggering multiple "False->True" edges and duplicate counting. Here use enter/exit hysteresis thresholds to mitigate jitter.
        # To prevent z-axis jitter near threshold from causing on/off flipping and duplicate counting:
        # Use enter/exit two sets of thresholds (hysteresis).
        # - enter strictly: entering target region counts as one-time landing
        # - exit loosely: small jitter within target region won't be misjudged as leaving
        swing_enter_distance_threshold = 0.03
        swing_exit_distance_threshold = 0.04  # >= enter
        swing_enter_z_threshold = 0.12
        swing_exit_z_threshold = 0.3  # >= enter

        if self._was_on_right:
            on_right = is_obj_swing_onto(
                self,
                obj=self.target_cube,
                target=self.target_right,
                distance_threshold=swing_exit_distance_threshold,
                z_threshold=swing_exit_z_threshold,
            )
        else:
            on_right = is_obj_swing_onto(
                self,
                obj=self.target_cube,
                target=self.target_right,
                distance_threshold=swing_enter_distance_threshold,
                z_threshold=swing_enter_z_threshold,
            )

        if self._was_on_left:
            on_left = is_obj_swing_onto(
                self,
                obj=self.target_cube,
                target=self.target_left,
                distance_threshold=swing_exit_distance_threshold,
                z_threshold=swing_exit_z_threshold,
            )
        else:
            on_left = is_obj_swing_onto(
                self,
                obj=self.target_cube,
                target=self.target_left,
                distance_threshold=swing_enter_distance_threshold,
                z_threshold=swing_enter_z_threshold,
            )
        if on_right:
            self.highlight_right_start = int(self.elapsed_steps[0].item())
            # Only accumulate swing count on "first" landing, avoid duplicate counting across continuous frames
            if not self._was_on_right:
                self.swing_count += 1
        if on_left:
            self.highlight_left_start = int(self.elapsed_steps[0].item())
            # Only accumulate swing count on "first" landing, avoid duplicate counting across continuous frames
            if not self._was_on_left:
                self.swing_count += 1

        # Update status after recording edge detection
        self._was_on_right = on_right
        self._was_on_left = on_left

        if self.swing_count > self.max_swings:
            if not self.swing_over_limit:
                # Print only once, warn swing count exceeded limit
                logger.debug(
                    f"Swing count exceeded: {self.swing_count}>{self.max_swings}"
                )
            self.swing_over_limit = True

        if self.highlight_right_start is not None:
            cur_step = int(self.elapsed_steps[0].item())

            highlight_obj(
                self,
                self.target_right,
                start_step=self.highlight_right_start,
                end_step=self.highlight_right_start + 20,
                cur_step=cur_step,
                disk_radius=self.cube_half_size * 2 * 1.003,
                disk_half_length=0.005 * 2 * 1.4,
                use_target_style=True,
                highlight_color=[1.0, 0.0, 0.0, 1.0],
            )
        if self.highlight_left_start is not None:
            cur_step = int(self.elapsed_steps[0].item())

            highlight_obj(
                self,
                self.target_left,
                start_step=self.highlight_left_start,
                end_step=self.highlight_left_start + 20,
                cur_step=cur_step,
                disk_radius=self.cube_half_size * 2 * 1.003,
                disk_half_length=0.005 * 2 * 1.4,
                use_target_style=True,
                highlight_color=[1.0, 0.0, 0.0, 1.0],
            )
        return obs, reward, terminated, truncated, info


from robomme_plus.registry import TASKS
from robomme_plus.tasks.counting.utils.environment import CountingMixin


@register_env(TASKS["SwingXtimes"].gym_id)
class SwingXtimes(CountingMixin, _TaskScene):
    """Swing a cube between evasive targets the requested number of cycles."""
    PERCEPTUAL_VARIATIONS = True

    task_name = "SwingXtimes"
