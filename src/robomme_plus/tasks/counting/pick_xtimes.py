"""Pick Xtimes: scene construction, native scoring and RoboMME-Plus task configuration."""

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
from robomme_plus.common.errors import SceneGenerationError
from robomme_plus.common.logging_utils import logger

# Robomme
from robomme_plus.simulation.actors import (
    build_button,
    spawn_random_cube,
    spawn_random_target,
)
from robomme_plus.control.planning import (
    solve_button,
    solve_pickup,
    solve_putonto_whenhold,
)
from robomme_plus.scoring.predicates import (
    is_any_obj_pickup,
    is_button_pressed,
    is_obj_dropped_onto,
    is_obj_pickup,
    sequential_task_check,
)
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
                                                                  'len(all_cubes), (1,))',
                                               'note': 'the earlier color draw is overridden by '
                                                       'the later target cube selection; keep the '
                                                       'original draw count and order'},
                'recovery': 'reuse the fail recover mode and the original inject_fail_grasp draw '
                            'given at entry',
                'task_expansion': 'repeatedly pick the same target_cube and place it on target, '
                                  'then press the button; non-targets are the complement of the '
                                  'candidates'},
 'positions': {'button': {'center_xy': [-0.2, 0],
                          'scale': 1.5,
                          'randomize_range_note': 'the original call site did not pass '
                                                  'randomize_range; keep the build_button '
                                                  'parameter default'},
               'cube_pose': {'half_size': 'self.cube_half_size',
                             'min_gap': 'self.cube_half_size',
                             'random_yaw': True,
                             'include_existing': False,
                             'include_goal': False},
               'target_pose': {'radius_factor': 2,
                               'thickness': 0.005,
                               'min_gap_factor': 2,
                               'include_existing': False,
                               'include_goal': False}},
 'decision': {'color': {'hard': 3, 'xhard1': 3},
              'number_range': {'hard': [4, 5], 'xhard1': [6, 6]},
              'target_cube_position_policy': {'region_center': [-0.1, 0], 'region_half_size': 0.2},
              'goal_position_policy': {'region_center': [-0.1, 0], 'region_half_size': 0.2},
              'distractor': None,
              'xhard1': {'target_cube_position_policy': {'region_center': [-0.1, 0],
                                                         'region_half_size': 0.25},
                         'goal_position_policy': {'region_center': [-0.1, 0],
                                                  'region_half_size': 0.2},
                         'distractor': {'colors': ['yellow'],
                                        'region_center': [-0.1, 0],
                                        'region_half_size': 0.25},
                         'min_center_dist_m': 0.08}}}


# Per-cube rejection budget for xhard cubes; 256 would leave about 3.5% of episodes unplaceable with the 8 cm center distance.
XHARD_CUBE_MAX_TRIALS = 1024


def _disk_avoid_obb(target, clearance):
    """Convert the target disk into a prebuilt OBB ``(center, axes, half extent)`` usable by cube rejection sampling.

    The disk is a visual-only actor (``add_collision=False``), so ``get_actor_obb`` cannot get a mesh and putting
    it into ``avoid`` would be silently ignored by ``spawn_random_cube`` (observed 2026-09-22). xhard places the
    disk before the cubes (G1), so its circumscribing square must be given explicitly: half extent = disk radius
    + disk gap - the cube's own gap, which makes the axis-wise test match the original disk-avoids-cube
    circle-to-box distance test and be more conservative along diagonals.
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


    # xhard4 (9 picks) only preserves the four-tier structure; do not use it for generation (exceeds the 1600-step limit).


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
        # Must run before any random number call and super().__init__(): an extra or missing draw here shifts all later draws
        self._sampling = copy.deepcopy(SCENE_CONFIG)
        self._spec = SpecRecorder(
            native_episode_spec,
            "PickXtimes",
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
        self.seed = seed
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

        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        button_cfg = self._sampling["positions"]["button"]
        button_obb = build_button(
            self,
            center_xy=tuple(button_cfg["center_xy"]),
            scale=button_cfg["scale"],
            generator=generator,
            recorder=self._spec,
            spec_path="layout.button_xy",
        )
        avoid = [button_obb]

        # The original tiers run the unchanged original code in _spawn_scene_objects_native; keep it identical.
        if is_newvalue_difficulty(self.difficulty):
            self._spawn_scene_objects_xhard(generator, avoid)
        else:
            self._spawn_scene_objects_native(generator, avoid)

            # Dynamically generate task list for N pickup-drop cycles
        tasks = []
        for i in range(self.num_repeats):
            tasks.append(
                {
                    "func": (lambda i=i: is_obj_pickup(self, obj=self.target_cube)),
                    "name": subgoal_language.get_subgoal_with_index(
                        i,
                        "pick up the {color} cube for the {idx} time",
                        color=self.target_color_name,
                    ),
                    "subgoal_segment": subgoal_language.get_subgoal_with_index(
                        i,
                        "pick up the {color} cube at <> for the {idx} time",
                        color=self.target_color_name,
                    ),
                    "choice_label": "pick up the cube",
                    "demonstration": False,
                    "failure_reason": ["wrong_cube", "premature_button"],
                    "failure_func": lambda: [
                        is_any_obj_pickup(self, self.non_target_cubes),
                        is_button_pressed(self, obj=self.button),
                    ],
                    "solve": lambda env, planner: solve_pickup(
                        env, planner, self.target_cube
                    ),
                    "segment": self.target_cube,
                }
            )
            tasks.append(
                {
                    "func": (
                        lambda: is_obj_dropped_onto(
                            self, obj=self.target_cube, target=self.target
                        )
                    ),
                    "name": f"place the {self.target_color_name} cube onto the target",
                    "subgoal_segment": f"place the {self.target_color_name} cube onto the target at <>",
                    "choice_label": "place the cube onto the target",
                    "demonstration": False,
                    "failure_reason": ["wrong_cube", "premature_button"],
                    "failure_func": lambda: [
                        is_any_obj_pickup(self, self.non_target_cubes),
                        is_button_pressed(self, obj=self.button),
                    ],
                    "solve": lambda env, planner: solve_putonto_whenhold(
                        env, planner, target=self.target
                    ),
                    "segment": self.target,
                }
            )

        tasks.append(
            {
                "func": lambda: is_button_pressed(self, obj=self.button),
                "name": "press the button to stop",
                "subgoal_segment": "press the button at <> to stop",
                "choice_label": "press the button to stop",
                "demonstration": False,
                "failure_reason": "extra_pickup",
                "failure_func": lambda: is_any_obj_pickup(self, self.all_cubes),
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

        # xhard distractors are a new draw and must come after all existing draws in this function (including the recovery draw).
        if is_newvalue_difficulty(self.difficulty):
            self._spawn_distractors_xhard(generator, avoid)

    def _spawn_scene_objects_native(self, generator, avoid):
        """Spawn cubes, disk and target cube for the original three tiers (the original middle part of ``_load_scene``, moved out verbatim; behavior unchanged)."""
        self.all_cubes = []  # Save all cube objects

        # Initialize storage for each color group
        self.red_cubes = []
        self.red_cube_names = []
        self.blue_cubes = []
        self.blue_cube_names = []
        self.green_cubes = []
        self.green_cube_names = []

        decision_cfg = self._sampling["decision"]
        cube_region = decision_cfg["target_cube_position_policy"]
        goal_region = decision_cfg["goal_position_policy"]
        cube_pose_cfg = self._sampling["positions"]["cube_pose"]
        target_pose_cfg = self._sampling["positions"]["target_pose"]
        cubes_per_color = self._sampling["parameters"]["cubes_per_color"]
        color_groups = [
            {
                "color": (1, 0, 0, 1),
                "name": "red",
                "list": self.red_cubes,
                "name_list": self.red_cube_names,
            },
            {
                "color": (0, 0, 1, 1),
                "name": "blue",
                "list": self.blue_cubes,
                "name_list": self.blue_cube_names,
            },
            {
                "color": (0, 1, 0, 1),
                "name": "green",
                "list": self.green_cubes,
                "name_list": self.green_cube_names,
            },
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

        # Generate 5 cubes for each color group
        for idx, group in enumerate(color_groups):
            if idx < decision_cfg["color"][self.difficulty]:
                for idx in range(cubes_per_color):
                    try:
                        cube = spawn_random_cube(
                            self,
                            color=group["color"],
                            avoid=avoid,
                            include_existing=False,
                            include_goal=False,
                            region_center=list(cube_region["region_center"]),
                            region_half_size=cube_region["region_half_size"],
                            half_size=self.cube_half_size,
                            min_gap=self.cube_half_size,
                            random_yaw=cube_pose_cfg["random_yaw"],
                            name_prefix=f"cube_{group['name']}_{idx}",
                            generator=generator,
                            recorder=self._spec,
                            spec_path=f"layout.cubes.{group['name']}_{idx}",
                        )
                    except RuntimeError as e:
                        logger.debug(
                            f"Failed to generate {group['name']} cube {idx}: {e}"
                        )
                        break

                    self.all_cubes.append(cube)
                    group["list"].append(cube)
                    cube_name = f"cube_{group['name']}_{idx}"
                    group["name_list"].append(cube_name)
                    setattr(self, cube_name, cube)
                    avoid.append(cube)

            logger.debug(f"Generated {len(group['list'])} {group['name']} cubes")

        logger.debug(
            f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})"
        )

        try:
            target = spawn_random_target(
                self,
                avoid=avoid,  # Use current avoidance list, containing all spawned cubes
                include_existing=False,  # Manually maintain list
                include_goal=False,  # Manually maintain list
                region_center=list(goal_region["region_center"]),
                region_half_size=goal_region["region_half_size"],
                radius=self.cube_half_size
                * target_pose_cfg["radius_factor"],  # Use radius instead of half_size
                thickness=target_pose_cfg["thickness"],  # target thickness
                min_gap=self.cube_half_size
                * target_pose_cfg["min_gap_factor"],  # Gap requirement same as cube
                name_prefix=f"target",
                generator=generator,
                recorder=self._spec,
                spec_path="layout.goal_xy",
            )
        except RuntimeError as e:
            logger.debug(f"Target sampling failed: {e}")

        # Assign target to self.target_0, self.target_1 etc. attributes
        setattr(self, f"target", target)
        # Add newly generated target to avoidance list
        avoid.append(target)

        # Randomly select one cube from all available cubes as the target
        if len(self.all_cubes) > 0:
            target_cube_idx = self._spec.value(
                "objects.target_cube_idx",
                torch.randint(0, len(self.all_cubes), (1,), generator=generator).item(),
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

    def _spawn_scene_objects_xhard(self, generator, avoid):
        """Spawn cubes, disk and target cube for xhard.

        Differences from the original path (all xhard only):
        * G1: **the disk is placed before the cubes**; cubes explicitly avoid it via ``_disk_avoid_obb``;
        * colors come from ``NATIVE_SAMPLING.parameters.color_pool`` (the single source of truth on the xhard
          path; no hardcoded literals);
        * V5: the three colored cubes are drawn uniformly in the region; the center distance between any two of
          the 6 cubes is at least ``min_center_dist_m`` (via ``spawn_random_cube(min_center_dist=...)``); placed
          cubes enter ``avoid`` as exact OBBs from ``cube_obb2d_exact`` (fixes degenerate obstacle boxes); each
          cube gets the budget ``XHARD_CUBE_MAX_TRIALS``;
        * D2: if the disk or a cube cannot be placed, raise ``SceneGenerationError`` directly instead of silently
          truncating or hitting an unbound variable;
        * the target cube is drawn from the explicit candidate list ``self.target_candidates`` (decoupled from
          ``all_cubes``, so distractors are never picked);
        * ``target_color_name`` is filled by object lookup instead of the three-color if chain (a new color
          would otherwise leave a stale value).
        Relative order of random calls: color_order -> target_color_idx -> disk -> cubes -> target_cube_idx,
        followed by the recovery action and the distractors.
        """
        xcfg = self._sampling["decision"][self.difficulty]
        cube_region = xcfg["target_cube_position_policy"]
        goal_region = xcfg["goal_position_policy"]
        min_center_dist = float(xcfg["min_center_dist_m"])
        cube_pose_cfg = self._sampling["positions"]["cube_pose"]
        target_pose_cfg = self._sampling["positions"]["target_pose"]
        cubes_per_color = self._sampling["parameters"]["cubes_per_color"]
        n_colors = self._sampling["decision"]["color"][self.difficulty]
        self._spec.record("layout.cube_min_center_dist", min_center_dist)
        self._xhard_cube_obbs = []

        self.all_cubes = []
        self.distractor_cubes = []
        self._cube_color_of = []
        color_groups = []
        for entry in self._sampling["parameters"]["color_pool"]:
            name = entry["name"]
            cube_list, name_list = [], []
            setattr(self, f"{name}_cubes", cube_list)
            setattr(self, f"{name}_cube_names", name_list)
            color_groups.append(
                {
                    "color": tuple(entry["rgba"]),
                    "name": name,
                    "list": cube_list,
                    "name_list": name_list,
                }
            )

        shuffle_indices = self._spec.value(
            "objects.color_order",
            torch.randperm(len(color_groups), generator=generator).tolist(),
        )
        color_groups = [color_groups[i] for i in shuffle_indices]
        # Same draw as the original path (its value is later overridden by the target cube color); keep it to preserve draw order
        target_color_idx = self._spec.value(
            "objects.target_color_idx",
            torch.randint(0, len(color_groups), (1,), generator=generator).item(),
        )
        self.target_color_name = color_groups[target_color_idx]["name"]

        disk_radius = self.cube_half_size * target_pose_cfg["radius_factor"]
        disk_gap = self.cube_half_size * target_pose_cfg["min_gap_factor"]
        try:
            target = spawn_random_target(
                self,
                avoid=avoid,
                include_existing=False,
                include_goal=False,
                region_center=list(goal_region["region_center"]),
                region_half_size=goal_region["region_half_size"],
                radius=disk_radius,
                thickness=target_pose_cfg["thickness"],
                min_gap=disk_gap,
                name_prefix="target",
                generator=generator,
                recorder=self._spec,
                spec_path="layout.goal_xy",
            )
        except RuntimeError as exc:
            raise SceneGenerationError(
                f"PickXtimes xhard: failed to sample the target disk placement: {exc}"
            ) from exc
        self.target = target
        avoid.append(target)
        avoid.append(
            _disk_avoid_obb(target, disk_radius + disk_gap - self.cube_half_size)
        )

        requested = min(n_colors, len(color_groups)) * cubes_per_color
        for group in color_groups[:n_colors]:
            for cube_idx in range(cubes_per_color):
                cube_name = f"cube_{group['name']}_{cube_idx}"
                try:
                    cube = spawn_random_cube(
                        self,
                        color=group["color"],
                        avoid=avoid,
                        include_existing=False,
                        include_goal=False,
                        region_center=list(cube_region["region_center"]),
                        region_half_size=cube_region["region_half_size"],
                        half_size=self.cube_half_size,
                        min_gap=self.cube_half_size,
                        random_yaw=cube_pose_cfg["random_yaw"],
                        name_prefix=cube_name,
                        generator=generator,
                        recorder=self._spec,
                        spec_path=f"layout.cubes.{group['name']}_{cube_idx}",
                        max_trials=XHARD_CUBE_MAX_TRIALS,
                        min_center_dist=(min_center_dist, self._xhard_cube_obbs),
                    )
                except RuntimeError as exc:
                    raise SceneGenerationError(
                        f"PickXtimes xhard: cube {cube_name} could not be placed: {exc}"
                    ) from exc
                self.all_cubes.append(cube)
                group["list"].append(cube)
                group["name_list"].append(cube_name)
                self._cube_color_of.append((cube, group["name"]))
                setattr(self, cube_name, cube)
                self._append_cube_obstacle_xhard(cube, avoid)
        self._spec.record(
            "objects.cube_count",
            {"requested": requested, "actual": len(self.all_cubes)},
        )
        if len(self.all_cubes) != requested or requested == 0:
            raise SceneGenerationError(
                f"PickXtimes xhard: colored cubes requested {requested}, got {len(self.all_cubes)}"
            )

        self.target_candidates = list(self.all_cubes)
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
        self.non_target_cubes = [
            cube for cube in self.all_cubes if cube is not self.target_cube
        ]

    def _color_name_of(self, cube):
        """Look up a cube's color name by object (xhard path); raise instead of leaving a stale value when it is not registered."""
        for actor, name in self._cube_color_of:
            if actor is cube:
                return name
        raise SceneGenerationError(
            "PickXtimes xhard: target cube is not in the color registry"
        )

    def _append_cube_obstacle_xhard(self, cube, avoid):
        """V5: register the just-placed cube as an exact OBB, as an obstacle and center-distance reference for later cubes.

        The actor itself is no longer put into ``avoid``: the actor path goes through ``_trimesh_box_to_obb2d``,
        which degenerates to a line segment for about 2/3 of cube orientations, so ``min_gap`` fails along its
        normal. Pure geometry; draws no random numbers.
        """
        obb = cube_obb2d_exact(cube, self.cube_half_size)
        self._xhard_cube_obbs.append(obb)
        avoid.append(obb)

    def _spawn_distractors_xhard(self, generator, avoid):
        """V4 xhard: place the distractor cubes, one per color in ``decision.distractor.colors`` (A5/B2).

        Distractors enter ``all_cubes`` and ``non_target_cubes`` (picking one wrongly triggers failure_func) but
        not ``target_candidates``. If one cannot be placed, raise ``SceneGenerationError`` directly (2.2(4); no
        silent truncation).
        V5: shares the center-distance rule, exact OBB obstacles and rejection budget with the colored cubes.
        """
        min_center_dist = float(
            self._sampling["decision"][self.difficulty]["min_center_dist_m"]
        )
        dcfg = self._sampling["decision"][self.difficulty]["distractor"]
        palette = {entry["name"]: entry["rgba"] for entry in BLOCK_DISTRACTOR_COLORS}
        names = list(dcfg["colors"])
        unknown = [name for name in names if name not in palette]
        if unknown:
            raise SceneGenerationError(
                f"PickXtimes xhard: distractor color not in BLOCK_DISTRACTOR_COLORS: {unknown}"
            )
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
                    random_yaw=self._sampling["positions"]["cube_pose"]["random_yaw"],
                    name_prefix=cube_name,
                    generator=generator,
                    recorder=self._spec,
                    spec_path=f"layout.distractors.{name}_0",
                    max_trials=XHARD_CUBE_MAX_TRIALS,
                    min_center_dist=(min_center_dist, self._xhard_cube_obbs),
                )
            except RuntimeError as exc:
                raise SceneGenerationError(
                    f"PickXtimes xhard: distractor cube {cube_name} could not be placed: {exc}"
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
            raise SceneGenerationError(
                f"PickXtimes xhard: distractor cubes requested {len(names)}, got {len(self.distractor_cubes)}"
            )
        # failure_func reads self.non_target_cubes at call time, so rebuilding it makes distractors count toward failure
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
            logger.debug(self.agent.robot.qpos)

    def evaluate(self, solve_complete_eval=False):

        previous_failure = getattr(self, "failureflag", None)
        self.successflag = torch.tensor([False])
        if previous_failure is not None and bool(previous_failure.item()):
            self.failureflag = previous_failure
        else:
            self.failureflag = torch.tensor([False])

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

        # highlight_obj(self,self.target_cube, start_step=0, end_step=30, cur_step=timestep)
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info


from robomme_plus.registry import TASKS
from robomme_plus.tasks.counting.utils.environment import CountingMixin
from robomme_plus.tasks.counting.utils.controllers import pick_gripper_open
from robomme_plus.tasks.counting.utils.motion import (
    PICK_LANDING_CLEARANCE_M,
    PICK_MIN_LIFT_M,
    ReleaseCounter,
)
from robomme_plus.common.sim import position, scalar, step_number


@register_env(TASKS["PickXtimes"].gym_id)
class PickXtimes(CountingMixin, _TaskScene):
    """Repeat fully lifted pick-and-place cycles onto the level's target."""
    PERCEPTUAL_VARIATIONS = True

    task_name = "PickXtimes"

    def _load_scene(self, *args, **kwargs):
        super()._load_scene(*args, **kwargs)
        if self._static_count is not None:
            self._static_native_checks = [entry["func"] for entry in self.task_list]

    def _initialize_episode(self, *args, **kwargs):
        self._static_placement = None
        super()._initialize_episode(*args, **kwargs)
        if self._static_count is None:
            return
        self._static_placement = ReleaseCounter(self.num_repeats)
        self._static_placement_events = []
        for index in range(2 * self.num_repeats):
            check = self._static_native_checks[index]
            if index % 2 == 0:
                self.task_list[index]["func"] = lambda _check=check: (
                    self._static_placement.lifted
                    and self._static_placement.was_grasped
                    and _check()
                )
            else:
                self.task_list[index]["func"] = lambda _check=check, _count=index // 2 + 1: (
                    self._static_placement.count >= _count and _check()
                )

    def evaluate(self, solve_complete_eval=False):
        counter = getattr(self, "_static_placement", None)
        if counter is not None:
            p = position(self.target_cube, initial=False)
            target = position(self.target, initial=False)
            change = counter.update(
                step_number(self),
                scalar(self.agent.is_grasping(self.target_cube)),
                p[2] >= self.cube_half_size + PICK_MIN_LIFT_M,
                np.linalg.norm(p[:2] - target[:2]) <= self.cube_half_size * 2,
                p[2] <= self.cube_half_size + PICK_LANDING_CLEARANCE_M,
                gripper_open=pick_gripper_open(self),
                gripper_z_m=float(position(self.agent.tcp, initial=False)[2]),
            )
            if change:
                self._static_placement_events.append({
                    "step": step_number(self), "kind": change,
                    "count": counter.count, "cube_xyz": p.tolist(),
                })
            if counter.failure:
                from robomme_plus.scoring.failures import record_failure
                record_failure(self, counter.failure)
                self.failureflag = torch.tensor([True], device=self.device)
        return super().evaluate(solve_complete_eval)


