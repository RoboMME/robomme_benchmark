"""Video Place Order: scene construction, native scoring and RoboMME-Plus task configuration."""

import copy
from typing import Dict, Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS

# Robomme
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

import robomme_plus.simulation.difficulty as difficulty_utils
import robomme_plus.control.reset as reset_panda
from robomme_plus.simulation.base import MemoryTask
from robomme_plus.simulation.difficulty import normalize_robomme_difficulty
from robomme_plus.simulation.specification import SpecRecorder
from robomme_plus.common.errors import SceneGenerationError
from robomme_plus.common.errors import (
    SceneGenerationError as _RealSceneGenerationError,
)
from robomme_plus.common.logging_utils import logger
from robomme_plus.simulation.actors import (
    build_button,
    spawn_random_cube,
)
from robomme_plus.control.planning import (
    solve_button,
    solve_hold_obj,
    solve_pickup,
    solve_reset,
    solve_strong_reset,
)
from robomme_plus.scoring.predicates import (
    is_any_obj_pickup,
    is_button_pressed,
    is_obj_dropped_onto,
    is_obj_dropped_onto_any,
    is_obj_pickup,
    reset_check,
    sequential_task_check,
    static_check,
)
from robomme_plus.simulation.choreography import swap_flat_two_lane
from robomme_plus.simulation.geometry import cube_obb2d_exact
from robomme_plus.tasks.reference.utils.home_sites import (
    build_home_sites,
    home_pose_record,
    validate_demo_plan,
)
from robomme_plus.tasks.reference.utils.place_swaps import (
    clocked_put as solve_putonto_whenhold,
)


SCENE_CONFIG = {'parameters': {'color_pool': [{'rgba': [1, 0, 0, 1], 'name': 'red'},
                               {'rgba': [0, 0, 1, 1], 'name': 'blue'},
                               {'rgba': [0, 1, 0, 1], 'name': 'green'}],
                'color_order': {'sampler': 'torch.randperm(3)'},
                'target_selection': {'sampler': 'torch.randint(0, len(all_cubes))'},
                'visit_selection': {'count_sampler': 'min(configured per-cube count, len(targets))',
                                    'order_sampler': 'torch.randperm(len(targets))[:count]'},
                'answer_selection': {'sampler': 'torch.randint(1, len(subset) + 1)',
                                     'mapping': 'target_target = visit_ids[which_in_subset - 1]'},
                'button_insertion': {'sampler': 'torch.randint(0, len(subset))',
                                     'mapping': 'button_task_index = k * 2 + 2'},
                'swap_selection': {'sampler': 'torch.randperm(len(targets))[:2]'},
                'swap_duration_steps': 50,
                'goal_site_z_override': -0.05,
                'recovery': 'reuses the fail recover mode given at the entry point and the '
                            'original generator',
                'target_slots': 4,
                'color': {'xhard1': 3}},
 'positions': {'goal': {'region_center': [-0.1, 0],
                        'region_half_size': 0.1,
                        'radius_factor': 5,
                        'thickness': 0.005},
               'button': {'center_xy': [0.1, 0], 'scale': 1.5, 'randomize_range': [0.05, 0.3]},
               'cubes': {'region_center': [0, 0], 'region_half_size': 0.2},
               'targets': {'region_center': [0, 0],
                           'region_half_size': 0.2,
                           'radius_factor': 2,
                           'thickness': 0.005}},
 'decision': {'demo_object_count': 1,
              'demo_return_policy': 'native_random_goal_site',
              'targets': {'xhard1': 4},
              'swap': {'xhard1': True},
              'xhard1': {'demo_object_count': 2,
                         'demo_return_policy': 'return_to_origin',
                         'visit_counts': [5, 5]}}}


def _is_newvalue_difficulty(value):
    """Use the shared difficulty-family predicate when available; otherwise fall back to a local set (supports targeted tests before the VP copy is merged)."""
    predicate = getattr(difficulty_utils, "is_newvalue_difficulty", None)
    if predicate is not None:
        return bool(predicate(value))
    return isinstance(value, str) and value.strip().lower() in {
        "xhard1",
        "xhard2",
        "xhard3",
        "xhard4",
    }


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
            "VideoPlaceOrder",
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
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

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
            # keep the difficulty selected by the seed instead of forcing it to easy

        self.onto_goalsite = False
        self.start_step = 99999
        self.end_step = 99999
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        self.generator.manual_seed(self.seed)
        # xhard raises the real exception class (retryable failure); the original difficulties keep the shadowed name unchanged.
        is_newvalue = _is_newvalue_difficulty(self.difficulty)
        _SceneGenError = (
            _RealSceneGenerationError
            if self.difficulty == "xhard" or is_newvalue
            else SceneGenerationError
        )

        try:
            self.table_scene = TableSceneBuilder(
                self, robot_init_qpos_noise=self.robot_init_qpos_noise
            )
            self.table_scene.build()

            yaw = 0
            rotate = np.array(
                [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            )  # Quaternion for z-axis rotation
            angles = torch.deg2rad(
                torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32)
            )  # (3,)
            rotate = matrix_to_quaternion(
                euler_angles_to_matrix(angles, convention="XYZ")
            )
            try:
                self.goal_site = self._spawn_target(
                    avoid=None,  # Use current avoidance list, containing all spawned cubes
                    include_existing=False,  # Manually maintain list
                    include_goal=False,  # Manually maintain list
                    region_center=list(
                        self._sampling["positions"]["goal"]["region_center"]
                    ),
                    region_half_size=self._sampling["positions"]["goal"][
                        "region_half_size"
                    ],
                    radius=self.cube_half_size
                    * self._sampling["positions"]["goal"][
                        "radius_factor"
                    ],  # Use radius instead of half_size
                    thickness=self._sampling["positions"]["goal"][
                        "thickness"
                    ],  # target thickness
                    min_gap=self.cube_half_size * 1,  # Gap requirement same as cube
                    name_prefix=f"goal_site",
                    generator=self.generator,
                    recorder=self._spec,
                    spec_path="layout.goal_xy",
                )
            except RuntimeError as exc:
                raise _SceneGenError("goal_site sampling failed") from exc
            avoid = []
            avoid.append(self.goal_site)
            button_cfg = self._sampling["positions"]["button"]
            cubes_cfg = self._sampling["positions"]["cubes"]
            targets_cfg = self._sampling["positions"]["targets"]
            decision_cfg = self._sampling["decision"]
            button_obb = build_button(
                self,
                center_xy=tuple(button_cfg["center_xy"]),
                scale=button_cfg["scale"],
                generator=self.generator,
                randomize_range=tuple(button_cfg["randomize_range"]),
                recorder=self._spec,
                spec_path="layout.button_xy",
            )
            avoid.append(button_obb)

            self.all_cubes = []  # Save all cube objects

            self.red_cubes = []
            self.red_cube_names = []
            self.blue_cubes = []
            self.blue_cube_names = []
            self.green_cubes = []
            self.green_cube_names = []

            cubes_per_color = 1
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
            self._spec.identity.setdefault(
                "difficulty", getattr(self, "difficulty", None)
            )
            shuffle_indices = self._spec.value(
                "objects.color_order",
                torch.randperm(len(color_groups), generator=self.generator).tolist(),
            )
            color_groups = [color_groups[i] for i in shuffle_indices]

            self.target_color_name = color_groups[0]["name"]
            logger.debug(f"Target color selected: {self.target_color_name}")

            for idx, group in enumerate(color_groups):
                if idx < self._sampling["parameters"]["color"][self.difficulty]:
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
                                random_yaw=True,
                                name_prefix=f"cube_{group['name']}_{cube_idx}",
                                generator=self.generator,
                                **self._xhard_spec_kwargs(
                                    f"layout.cubes.{group['name']}_{cube_idx}"
                                ),
                            )
                        except RuntimeError as exc:
                            raise _SceneGenError(
                                f"Failed to generate {group['name']} cube {cube_idx}: {exc}"
                            ) from exc

                        self.all_cubes.append(cube)
                        group["list"].append(cube)
                        cube_name = f"cube_{group['name']}_{cube_idx}"
                        group["name_list"].append(cube_name)
                        setattr(self, cube_name, cube)
                        avoid.append(
                            cube_obb2d_exact(cube.initial_pose, self.cube_half_size)
                        )

                logger.debug(f"Generated {len(group['list'])} {group['name']} cubes")

            logger.debug(
                f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})"
            )

            self.targets = []
            for i in range(int(decision_cfg["targets"][self.difficulty])):
                if i < decision_cfg["targets"][self.difficulty]:
                    try:
                        target = self._spawn_target(
                            avoid=avoid,  # Use current avoidance list, containing all spawned cubes
                            include_existing=False,  # Manually maintain list
                            include_goal=False,  # Manually maintain list
                            region_center=list(targets_cfg["region_center"]),
                            region_half_size=targets_cfg["region_half_size"],
                            radius=self.cube_half_size
                            * targets_cfg[
                                "radius_factor"
                            ],  # Use radius instead of half_size
                            thickness=targets_cfg["thickness"],  # target thickness
                            min_gap=self.cube_half_size
                            * 1,  # Gap requirement same as cube
                            name_prefix=f"target_{i}",
                            generator=self.generator,
                            **self._xhard_spec_kwargs(f"layout.targets.{i}"),
                        )
                    except RuntimeError as exc:
                        raise _SceneGenError(
                            f"Target {i + 1} sampling failed: {exc}"
                        ) from exc

                    self.targets.append(target)
                    setattr(self, f"target_{i}", target)
                    avoid.append(target)

            self._load_scene_xhard_tail()
            return

        except _SceneGenError:
            raise
        except Exception as exc:
            raise _SceneGenError(
                f"Failed to load VideoPlaceOrder scene for seed {self.seed}"
            ) from exc

    # ------------------------------------------------------------------
    # xhard: demo two cubes, each put back at its original position; everything else unchanged
    # ------------------------------------------------------------------
    def _xhard_spec_kwargs(self, spec_path):
        """Pass the recorder to spawn_* only for xhard; the original difficulties return an empty dict (identical call shape)."""
        pass
        return {"recorder": self._spec, "spec_path": spec_path}

    @staticmethod
    def xhard_button_task_index(visit_counts, button_after_visit):
        """Index at which the button is inserted into the demo sequence (generalizes the original ``k*2+2``).

        The xhard demo sequence is expanded object by object as
        ``[the object's visit pick+drop pairs ..., return-to-origin pick+drop]``, and every unit is 2 tasks.
        The button goes after the "global ``button_after_visit + 1``-th visit placement", i.e. between two units
        and never between a pick and its drop:

            index = 2 × (button_after_visit + 1 + number of return-to-origin units already completed before it)

        When an object's last visit is exactly the insertion point, the button comes before that object's return
        to origin. With ``visit_counts=[n]`` this reduces to the original formula ``k*2+2``.
        """
        placed = int(button_after_visit) + 1
        if not 1 <= placed <= sum(visit_counts):
            raise ValueError(
                f"button_after_visit={button_after_visit} exceeds total visit count {sum(visit_counts)}"
            )
        homes_before = 0
        cumulative = 0
        for count in visit_counts:
            cumulative += int(count)
            if cumulative < placed:
                homes_before += 1
        return 2 * (placed + homes_before)

    def _load_scene_xhard_tail(self):
        """Choose each cube's ordered visits and a uniformly sampled requested visit.

        Each cube visits five distinct rings, capped by the available ring count
        (four on easy), then returns home. Hard retains both cubes' histories;
        easy and medium later retain only the requested cube's history. The home
        return is separate from the numbered target placements.
        """
        decision_cfg = self._sampling["decision"]
        tier_key = "xhard" if self.difficulty == "xhard" else self.difficulty
        xhard_cfg = decision_cfg[tier_key]
        is_v6_tier = _is_newvalue_difficulty(self.difficulty)
        demo_count, return_policy = validate_demo_plan(
            xhard_cfg["demo_object_count"],
            xhard_cfg["demo_return_policy"],
            self.difficulty,
            len(self.all_cubes),
        )
        configured_counts = [
            min(int(value), len(self.targets)) for value in xhard_cfg["visit_counts"]
        ]
        if (
            len(configured_counts) != demo_count
            or any(
                value < 2 or value > len(self.targets)
                for value in configured_counts
            )
        ):
            raise _RealSceneGenerationError(
                f"VideoPlaceOrder {self.difficulty} visit_counts={configured_counts} is invalid; "
                f"requires {demo_count} counts between 2 and {len(self.targets)}"
            )
        count_order = (
            torch.randperm(demo_count, generator=self.generator).tolist()
            if len(set(configured_counts)) > 1
            else list(range(demo_count))
        )
        counts_drawn = [configured_counts[index] for index in count_order]
        visit_counts = [
            int(value)
            for value in self._spec.value(
                "objects.visit_counts_by_object",
                counts_drawn,
                decision_key=f"{tier_key}.visit_counts",
            )
        ]
        if sorted(visit_counts) != sorted(configured_counts):
            raise _RealSceneGenerationError(
                f"VideoPlaceOrder replayed visit_counts={visit_counts} does not match configured multiset {configured_counts}"
            )
        if len(self.targets) < 2:
            raise _RealSceneGenerationError(
                f"VideoPlaceOrder xhard needs at least 2 targets, got {len(self.targets)}"
            )

        demo_ids = self._spec.value(
            "objects.demo_ids",
            torch.randperm(len(self.all_cubes), generator=self.generator)[
                :demo_count
            ].tolist(),
            decision_key=f"{tier_key}.demo_object_count",
        )
        demo_ids = [int(i) for i in demo_ids]
        if len(demo_ids) != demo_count:
            raise _RealSceneGenerationError(
                f"requested {demo_count} demo cubes, got {len(demo_ids)}"
            )
        self.demo_cubes = [self.all_cubes[i] for i in demo_ids]

        self.swap_target_a = None
        self.swap_target_b = None
        self.swap_target_other = []
        if decision_cfg["swap"][self.difficulty] == True:
            perm = self._spec.value(
                "objects.swap_pair_ids",
                torch.randperm(len(self.targets), generator=self.generator).tolist(),
                decision_key=f"swap.{self.difficulty}",
            )
            swap_idx_a, swap_idx_b = int(perm[0]), int(perm[1])
            self.swap_target_a = self.targets[swap_idx_a]
            self.swap_target_b = self.targets[swap_idx_b]
            self.swap_target_other = [
                target
                for idx, target in enumerate(self.targets)
                if idx not in (swap_idx_a, swap_idx_b)
            ]

        self.demo_visit_targets = []
        for k in range(demo_count):
            count = visit_counts[k]
            ids = self._spec.value(
                f"objects.visit_ids_by_object.{k}",
                torch.randperm(len(self.targets), generator=self.generator)[
                    :count
                ].tolist(),
                decision_key=f"{tier_key}.visit_counts"
                if is_v6_tier
                else "xhard.demo_object_count",
            )
            self.demo_visit_targets.append([self.targets[int(i)] for i in ids])
            pass

        answer_index = int(
            self._spec.value(
                "objects.answer_demo_index",
                torch.randint(0, demo_count, (1,), generator=self.generator).item(),
                decision_key=f"{tier_key}.demo_object_count",
            )
        )
        self.target_cube = self.demo_cubes[answer_index]
        for color_name, group in (
            ("red", self.red_cubes),
            ("blue", self.blue_cubes),
            ("green", self.green_cubes),
        ):
            if self.target_cube in group:
                self.target_color_name = color_name
        self.non_target_cubes = [
            cube for cube in self.all_cubes if cube != self.target_cube
        ]
        # Same name as the original difficulties, so downstream readers (task_goal etc.) keep working.
        self.which_targets_to_pick = self.demo_visit_targets[answer_index]
        self.which_in_subset = int(
            self._spec.value(
                "objects.which_in_subset",
                torch.randint(
                    1,
                    len(self.which_targets_to_pick) + 1,
                    (1,),
                    generator=self.generator,
                ).item(),
                decision_key=f"{tier_key}.visit_counts"
                if is_v6_tier
                else "xhard.demo_object_count",
            )
        )
        self.target_target = self.which_targets_to_pick[self.which_in_subset - 1]
        self.targets_not_true = [t for t in self.targets if t != self.target_target]
        self._spec.record(
            "actions.target_target_id", self.targets.index(self.target_target)
        )

        button_after_visit = int(
            self._spec.value(
                "objects.button_after_visit_index",
                torch.randint(
                    0, sum(visit_counts), (1,), generator=self.generator
                ).item(),
                decision_key=f"{tier_key}.visit_counts"
                if is_v6_tier
                else "xhard.demo_object_count",
            )
        )
        self.button_task_index = self.xhard_button_task_index(
            visit_counts, button_after_visit
        )
        self._spec.record("actions.button_task_index", self.button_task_index)
        self.target_placement_count = sum(visit_counts)
        self._spec.record("actions.target_placement_count", sum(visit_counts))

        # Must run after all spawns; builds sites from each cube's initial pose directly (not via spawn_random_target).
        self._build_xhard_final_sites(return_policy)

    def _build_xhard_final_sites(self, return_policy):
        """Return every demonstrated cube to its initial pose."""
        self.xhard_home_sites, self._xhard_home_checks = build_home_sites(
            self, self.demo_cubes, self.generator
        )
        self._spec.record(
            "actions.return_pose_by_object_id",
            home_pose_record(self.demo_cubes, self.xhard_home_sites),
        )
        self.xhard_goal_drop_sites = []
        self._xhard_final_sites = [
            (c, h, "home") for c, h in zip(self.demo_cubes, self.xhard_home_sites)
        ]
        return

    def _build_xhard_task_list(self):
        """Task list for xhard (rebuilt on every _initialize_episode, same timing as the original difficulties)."""
        pair_tasks = []
        for (cube, site, kind), visits in zip(
            self._xhard_final_sites, self.demo_visit_targets
        ):
            for target in visits:
                pair_tasks.extend(self._xhard_pick_place(cube, target))
            pair_tasks.extend(self._xhard_pick_place(cube, site, home=kind))

        button_task = {
            "func": (lambda: is_button_pressed(self, obj=self.button)),
            "name": "press the button",
            "subgoal_segment": f"press the button at <>",
            "choice_label": "press the button",
            "demonstration": True,
            "failure_func": None,
            "solve": lambda env, planner: solve_button(env, planner, self.button),
            "segment": self.cap_link,
        }
        tasks = (
            pair_tasks[: self.button_task_index]
            + [button_task]
            + pair_tasks[self.button_task_index :]
        )
        tasks.extend(
            [
                {
                    "func": lambda: static_check(
                        self, timestep=int(self.elapsed_steps), static_steps=20
                    ),
                    "name": "static",
                    "subgoal_segment": f"static",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [
                        solve_reset(env, planner),
                        solve_hold_obj(env, planner, static_steps=20),
                    ],
                },
                {
                    "func": lambda: static_check(
                        self, timestep=int(self.elapsed_steps), static_steps=60
                    ),
                    "name": "static",
                    "subgoal_segment": f"static",
                    "specialflag": "swap",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [
                        solve_hold_obj(env, planner, static_steps=60)
                    ],
                },
                {
                    "func": lambda: reset_check(self),
                    "name": "NO RECORD",
                    "subgoal_segment": f"NO RECORD",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [solve_strong_reset(env, planner)],
                },
                {
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                    "name": f"pick up the cube",
                    "subgoal_segment": f"pick up the cube at <>",
                    "choice_label": "pick up the cube",
                    "demonstration": False,
                    "failure_reason": "wrong_cube",
                    "failure_func": lambda: is_any_obj_pickup(
                        self, self.non_target_cubes
                    ),
                    "solve": lambda env, planner: [
                        solve_pickup(env, planner, obj=self.target_cube)
                    ],
                    "segment": self.target_cube,
                },
                {
                    "func": (
                        lambda: is_obj_dropped_onto(
                            self, obj=self.target_cube, target=self.target_target
                        )
                    ),
                    "name": "place the cube onto the correct target",
                    "subgoal_segment": f"place the cube onto the correct target at <>",
                    "choice_label": "drop onto",
                    "demonstration": False,
                    "failure_reason": "wrong_placement",
                    "failure_func": (
                        lambda: is_obj_dropped_onto_any(
                            self, obj=self.target_cube, target=self.targets_not_true
                        )
                    ),
                    "solve": lambda env, planner: [
                        solve_putonto_whenhold(env, planner, target=self.target_target)
                    ],
                    "segment": self.target_target,
                },
            ]
        )

        self.task_list = tasks
        self.fail_grasp_task_index = None

    def _xhard_pick_place(self, cube, target, home=False):
        """One pick + drop pair of a demo segment (the closure binds the current cube and site via default args).

        ``home``: False = drop on the target; True / "home" = put back at the original position;
        "goal" = not put back, dropped onto the table (goal_site region, text reuses the original
        difficulties' "drop the cube onto table").
        """
        if home == "goal":
            name = "drop the cube onto table"
            segment_text = "drop the cube onto table"
        elif home is True or home == "home":
            name = "put the cube back to its original position"
            # Home sites are hidden, so no coordinates can be filled in; V5 xhard keeps the old ``at <>`` text.
            segment_text = (
                name
                if _is_newvalue_difficulty(self.difficulty)
                else "put the cube back to its original position at <>"
            )
        else:
            name = "drop the cube onto target"
            segment_text = "drop the cube onto target at <>"
        return [
            {
                "func": (lambda cube=cube: is_obj_pickup(self, obj=cube)),
                "name": f"pick up the cube",
                "subgoal_segment": f"pick up the cube at <>",
                "choice_label": "pick up the cube",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner, cube=cube: solve_pickup(
                    env, planner, obj=cube
                ),
                "segment": cube,
            },
            {
                "func": (
                    lambda cube=cube, target=target: is_obj_dropped_onto(
                        self, obj=cube, target=target
                    )
                ),
                "name": name,
                "subgoal_segment": segment_text,
                "choice_label": "drop onto",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner, target=target: solve_putonto_whenhold(
                    env, planner, target=target
                ),
                "segment": target,
            },
        ]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._native_init_index = getattr(self, "_native_init_index", -1) + 1
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos = reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)

            pose_p = self.goal_site.pose.p.tolist()[0]
            pose_q = self.goal_site.pose.q.tolist()[0]
            pose_p[2] = -0.05
            self.goal_site.set_pose(sapien.Pose(p=pose_p, q=pose_q))
            # print(self.goal_site.pose.p)

            self._build_xhard_task_list()
            return


    def evaluate(self, solve_complete_eval=False):

        self.successflag = torch.tensor([False])
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
            self.current_task_specialflags,
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

        # highlight_obj(self,self.target_cube, start_step=0, end_step=100, cur_step=timestep)

        if self.current_task_specialflag == "swap":
            if self.onto_goalsite == False:
                self.onto_goalsite = True
                self.start_step = int(self.elapsed_steps.item())
                self.end_step = int(self.elapsed_steps.item()) + 50

        if self.swap_target_a is not None and self.swap_target_b is not None:
            other_bins = self.swap_target_other if self.swap_target_other else None
            swap_flat_two_lane(
                self,
                cube_a=self.swap_target_a,
                cube_b=self.swap_target_b,
                start_step=self.start_step,
                end_step=self.end_step,
                cur_step=self.elapsed_steps,
                lane_offset=0.1,
                smooth=True,
                keep_upright=True,
                other_cube=other_bins,
            )
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info


from robomme_plus.registry import TASKS
from robomme_plus.tasks.reference.utils.place_environment import (
    VideoPlaceMixin,
)


@register_env(TASKS["VideoPlaceOrder"].gym_id)
class VideoPlaceOrder(VideoPlaceMixin, _TaskScene):
    """Remember a requested cube's ordinal target through moving-ring demonstrations."""
    PERCEPTUAL_VARIATIONS = True

    task_name = "VideoPlaceOrder"
