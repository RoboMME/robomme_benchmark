"""Pick Highlight: scene construction, native scoring and RoboMME-Plus task configuration."""

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
)
from robomme_plus.control.planning import (
    solve_button,
    solve_pickup,
    solve_putdown_whenhold,
)
from robomme_plus.scoring.predicates import (
    is_any_obj_pickup,
    is_any_obj_pickup_flag_currentpickup,
    is_button_pressed,
    is_obj_dropped_currentpickup,
    is_obj_pickup,
    sequential_task_check,
)
from robomme_plus.common.errors import SamplingConfigError
from robomme_plus.simulation.choreography import highlight_obj
from robomme_plus.tasks.reference.utils.highlight_timing import (
    highlight_schedule,
    resolve_highlight_mode,
)
from robomme_plus.tasks.reference.utils.highlight_levels import (
    COMPACT_CUBE_BOUNDS,
    DENSE_CUBE_BOUNDS,
    DENSE_CUBE_GAP_FACTOR,
    DURATION_STEPS,
    cube_count_range,
    sample_highlight_ids,
    sample_similar_colors,
    validate_color_mode,
)
from robomme_plus.tasks.reference.utils.table_layout import TABLE_BOUNDS, sample_cube_pool


# xhard-only decision entries; assert_native_decision requires the original tiers to be identical once xhard is removed.
from robomme_plus.simulation.geometry import (
    cube_obb2d_exact,
    hsv_floor_rgb,
)


SCENE_CONFIG = {'parameters': {'color_pool': [{'rgba': [1, 0, 0, 1], 'name': 'red'},
                               {'rgba': [0, 0, 1, 1], 'name': 'blue'},
                               {'rgba': [0, 1, 0, 1], 'name': 'green'}],
                'color_draw': {'sampler': 'torch.randint',
                               'low': 0,
                               'high_exclusive': 'len(color_pool)'},
                'target_selection': {'sampler': 'torch.randperm(len(all_cubes))[:pickup]'},
                'spawn_failure': 'on spawn failure break, keep the actual count, no redraw',
                'recovery': 'reuse the fail recover mode and original generator given at entry'},
 'positions': {'button': {'center_xy': [-0.2, 0], 'scale': 1.5},
               'cubes': {'region_center': [-0.1, 0],
                         'region_half_size': 0.2,
                         'min_gap_factor': 2,
                         'random_yaw': True,
                         'include_existing': False,
                         'include_goal': False},
               'highlight_window': {'start_step': 10,
                                    'end_step': 119}},
 'decision': {'layout_mode': 'native_region',
              'cube_region': {'region_center': [-0.1, 0], 'region_half_size': 0.2},
              'highlight_count': {'xhard4': [5, 9],
                                  'xhard1': [3, 5],
                                  'xhard2': [4, 6],
                                  'xhard3': [5, 9]},
              'block_color_policy': 'native_per_cube_uniform',
              'xhard4': {'block_color_policy': 'hsv_floor',
                         'block_color_hsv': {'h_range': [0.0, 1.0],
                                             's_range': [0.5, 1.0],
                                             'v_range': [0.4, 1.0]},
                         'subgoal_color_suffix': 'omit'},
              'xhard1': {'block_color_policy': 'hsv_floor',
                         'block_color_hsv': {'h_range': [0.0, 1.0],
                                             's_range': [0.5, 1.0],
                                             'v_range': [0.4, 1.0]},
                         'subgoal_color_suffix': 'omit'},
              'xhard2': {'block_color_policy': 'hsv_floor',
                         'block_color_hsv': {'h_range': [0.0, 1.0],
                                             's_range': [0.5, 1.0],
                                             'v_range': [0.4, 1.0]},
                         'subgoal_color_suffix': 'omit'},
              'xhard3': {'block_color_policy': 'hsv_floor',
                         'block_color_hsv': {'h_range': [0.0, 1.0],
                                             's_range': [0.5, 1.0],
                                             'v_range': [0.4, 1.0]},
                         'subgoal_color_suffix': 'omit'}}}


def _closed_range(value, key):
    """Validate an xhard closed interval ``[lo, hi]`` as two integers; reject malformed external configs directly."""
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)
        or value[0] < 1
        or value[0] > value[1]
    ):
        raise SamplingConfigError(
            f"PickHighlight: decision.{key} must be a closed interval [lo, hi] (integers with 1<=lo<=hi), got {value!r}"
        )
    return int(value[0]), int(value[1])


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
        # Must run before any random number call and super().__init__()
        self._sampling = copy.deepcopy(SCENE_CONFIG)
        self._sampling["positions"]["highlight_window"]["duration_steps"] = DURATION_STEPS[self.level]
        self._sampling["positions"]["highlight_window"]["mode"] = self.highlight_mode
        self._spec = SpecRecorder(
            native_episode_spec,
            "PickHighlight",
            {"seed": seed},
            difficulty=kwargs.get("difficulty"),
        )
        self._native_init_index = -1
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.use_demonstrationwrapper = False
        self.demonstration_record_traj = False
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
        self.generator.manual_seed(self.seed)

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

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        self.generator.manual_seed(self.seed)
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        button_cfg = self._sampling["positions"]["button"]
        cubes_cfg = self._sampling["positions"]["cubes"]
        decision_cfg = self._sampling["decision"]
        cube_region = decision_cfg["cube_region"]
        self._spec.identity.setdefault("difficulty", getattr(self, "difficulty", None))
        button_obb = build_button(
            self,
            center_xy=tuple(button_cfg["center_xy"]),
            scale=button_cfg["scale"],
            generator=self.generator,
            recorder=self._spec,
            spec_path="layout.button_xy",
        )
        avoid = [button_obb]

        self.all_cubes = []  # Save all cube objects
        self.all_cube_names = []
        self.all_cube_colors = []

        # List of available colors
        # Pool comes from the snapshot; order must match the original literal (red, blue, green)
        available_colors = [
            {"color": tuple(entry["rgba"]), "name": entry["name"]}
            for entry in self._sampling["parameters"]["color_pool"]
        ]

        xhard = is_newvalue_difficulty(self.difficulty)
        tier = self.difficulty
        xhard_cfg = decision_cfg[tier]
        if xhard_cfg["block_color_policy"] != "hsv_floor":
            raise SamplingConfigError(
                f"PickHighlight: decision.{tier}.block_color_policy only supports 'hsv_floor', "
                f"got {xhard_cfg['block_color_policy']!r}"
            )
        if xhard_cfg["subgoal_color_suffix"] != "omit":
            raise SamplingConfigError(
                f"PickHighlight: decision.{tier}.subgoal_color_suffix currently only implements 'omit', "
                f"got {xhard_cfg['subgoal_color_suffix']!r}"
            )
        spawn_lo, spawn_hi = cube_count_range(self.level, self.layout_version)
        highlight_lo, highlight_hi = _closed_range(
            decision_cfg["highlight_count"][tier], f"highlight_count.{tier}"
        )
        if spawn_lo < highlight_hi:
            raise SamplingConfigError(
                f"PickHighlight: {tier} requires spawn lower bound >= highlight upper bound, got spawn=[{spawn_lo},{spawn_hi}] "
                f"highlight=[{highlight_lo},{highlight_hi}]"
            )
        # New sampling point (xhard only): must come before the cube loop because it sets the loop length
        num_cubes_to_spawn = int(
            self._spec.value(
                "objects.n_cubes",
                int(
                    torch.randint(
                        spawn_lo, spawn_hi + 1, (1,), generator=self.generator
                    ).item()
                ),
                decision_key=f"spawn_count.{tier}",
            )
        )

        button_xy = self.button.pose.p.detach().cpu().numpy().reshape(-1, 3)[0, :2]
        bounds = COMPACT_CUBE_BOUNDS if self.layout_version >= 3 else TABLE_BOUNDS
        gap_factor = cubes_cfg["min_gap_factor"]
        if self.layout_version >= 4:
            bounds = DENSE_CUBE_BOUNDS
            gap_factor = DENSE_CUBE_GAP_FACTOR
        gap = self.cube_half_size * gap_factor
        pool_xy, pool_yaw = sample_cube_pool(
            int(self.seed) + 872433, num_cubes_to_spawn, self.cube_half_size,
            gap,
            obstacles=[(button_xy, np.sqrt(2) * 0.025 * button_cfg["scale"])],
            bounds=bounds,
        )
        self._spec.record("custom_cube_layout", {
            "version": max(2, self.layout_version),
            "policy": "shared reachable pool; whole-layout restarts; shuffled roles",
            "bounds_xy_m": np.asarray(bounds).tolist(),
        })
        if self.layout_version >= 4:
            self._spec.record("custom_cube_layout.min_gap_m", gap)
            self._spec.record("custom_cube_layout.target_selection", "uniform subset after placement")
        similar_colors = None
        if self.color_mode == "similar":
            similar_colors, palette = sample_similar_colors(self.seed, num_cubes_to_spawn)
            self._spec.record("objects.color_palette", palette)
        for cube_idx in range(num_cubes_to_spawn):
            # Retain these draws so color changes cannot alter goal counts or
            # other choices on the legacy torch stream.
            independent_rgba = hsv_floor_rgb(
                torch.rand(3, generator=self.generator).tolist(),
                xhard_cfg["block_color_hsv"],
            ) + [1.0]
            rgba = self._spec.value(
                f"objects.color_rgba.{cube_idx}",
                similar_colors[cube_idx] if similar_colors is not None else independent_rgba,
                decision_key=f"{tier}.block_color_policy",
            )
            chosen_color = {
                "color": tuple(float(c) for c in rgba),
                "name": "rgb",
                "label": None,
            }

            try:
                x, y, yaw = self._spec.value(
                    f"layout.cubes.{cube_idx}", [*pool_xy[cube_idx], pool_yaw[cube_idx]]
                )
                cube = spawn_random_cube(
                    self,
                    color=chosen_color["color"],
                    avoid=avoid,
                    include_existing=False,
                    include_goal=False,
                    region_center=list(cube_region["region_center"]),
                    region_half_size=cube_region["region_half_size"],
                    half_size=self.cube_half_size,
                    min_gap=gap,
                    random_yaw=cubes_cfg["random_yaw"],
                    name_prefix=f"cube_{chosen_color['name']}_{cube_idx}",
                    generator=self.generator,
                    recorder=self._spec,
                    spec_path=f"layout.cubes.{cube_idx}",
                    fixed_xy=[x, y],
                    fixed_yaw=yaw,
                )

                cube_name = f"cube_{chosen_color['name']}_{cube_idx}"

                # Add cube immediately after successful creation
                self.all_cubes.append(cube)
                self.all_cube_names.append(cube_name)
                self.all_cube_colors.append(
                    chosen_color.get("label", chosen_color["name"])
                )
                setattr(self, cube_name, cube)
                avoid.append(
                    cube_obb2d_exact(cube.initial_pose, self.cube_half_size)
                )

            except RuntimeError as e:
                raise SceneGenerationError(
                    f"PickHighlight xhard: could not place all cubes, requested {num_cubes_to_spawn}, got {len(self.all_cubes)}"
                    f" (cube {cube_idx} failed: {e})"
                ) from e

        logger.debug(f"Generated {len(self.all_cubes)} cubes total")

        self._spec.record("objects.n_cubes_spawned", len(self.all_cubes))
        # Preserve recorded episodes' random stream. New episodes select on a
        # separate stream after the full pool exists, without using geometry.
        if self.layout_version < 4:
            permutation = torch.randperm(
                len(self.all_cubes), generator=self.generator
            ).tolist()
        highlight_count = int(
            self._spec.value(
                "objects.highlight_count",
                int(
                    torch.randint(
                        highlight_lo,
                        highlight_hi + 1,
                        (1,),
                        generator=self.generator,
                    ).item()
                ),
                decision_key=f"highlight_count.{tier}",
            )
        )
        # randperm(len)[:k] silently truncates when k > len, so reject explicitly
        if highlight_count > len(self.all_cubes):
            raise SceneGenerationError(
                f"PickHighlight xhard: highlight count {highlight_count} exceeds the actual cube count {len(self.all_cubes)}"
            )
        target_cube_indices = self._spec.value(
            "objects.highlight_ids",
            (sample_highlight_ids(self.seed, len(self.all_cubes), highlight_count)
             if self.layout_version >= 4 else permutation[:highlight_count]),
            decision_key=f"highlight_count.{tier}",
        )

        self.target_cubes = [self.all_cubes[idx] for idx in target_cube_indices]
        self.target_cube_names = [
            self.all_cube_names[idx] for idx in target_cube_indices
        ]
        window = self._sampling["positions"]["highlight_window"]
        self.highlight_schedule = highlight_schedule(
            self.seed,
            len(self.target_cubes),
            start_step=window["start_step"],
            end_step=window["end_step"],
            duration_steps=window["duration_steps"],
            mode=self.highlight_mode,
        )
        # Highlight end steps are inclusive. Movement may begin only on the
        # following step, once step() has removed the final visible cue.
        self.local_adjustment_start_step = max(t["end_step"] for t in self.highlight_schedule) + 1
        self._spec.record("actions.highlight_mode", self.highlight_mode)
        self._spec.record(
            "actions.highlight_schedule",
            [
                {"cube": name, **timing}
                for name, timing in zip(self.target_cube_names, self.highlight_schedule)
            ],
        )
        self.target_cube_colors = [
            self.all_cube_colors[idx] for idx in target_cube_indices
        ]
        self.target_labels = [
            (color or name or "target")
            for color, name in zip(self.target_cube_colors, self.target_cube_names)
        ]
        if self.local_adjustments:
            from robomme_plus.tasks.reference.utils.highlight_motion import (
                DURATION, INTERVAL_PER_LEVEL, MIN_DISPLACEMENT, CLUSTERS_PER_LEVEL,
                CLUSTER_SIZE_PER_LEVEL,
                START_CLEARANCE, STOP_CLEARANCE,
            )

            self._spec.record("actions.local_adjustments", {
                "cluster_size": list(CLUSTER_SIZE_PER_LEVEL[self.level]), "duration_steps": DURATION,
                "minimum_displacement_m": MIN_DISPLACEMENT,
                "destinations": "new local positions; every cube changes relative sides",
                "trigger": "repeatedly after highlights end; seeded batches by difficulty",
                "first_possible_step": self.local_adjustment_start_step,
                "requested_clusters_per_batch": list(CLUSTERS_PER_LEVEL[self.level]),
                "batch_policy": "equal probability; fewer groups if visibility or clearance prevents movement",
                "max_active_clusters": CLUSTERS_PER_LEVEL[self.level][1],
                "selection": "seeded weighted anchors; partner combinations weighted by product of member weights",
                "cube_selection_weight": "1 + batches skipped since that cube last started moving; reset on movement and each episode",
                "event_limit": None, "stationary_interval_steps": INTERVAL_PER_LEVEL[self.level],
                "eligible_cubes": "resting, unheld cubes, including previously picked cubes",
                "start_gripper_xy_clearance_m": START_CLEARANCE,
                "stop_gripper_xy_clearance_m": STOP_CLEARANCE,
                "arm_visibility": {"camera": "base_camera", "maximum_occluded_fraction": .3,
                                   "check": "every movement step; current and next cube poses",
                                   "blocked_group": "stop in place"},
            })
        self._build_task_list()

    def _remaining_highlight_cubes(self):
        return [cube for cube in self.target_cubes if cube not in self._highlight_picks]

    def _accept_highlight_pickup(self, stage):
        if len(self._highlight_picks) > stage:
            return True
        if len(self._highlight_picks) != stage:
            return False
        if is_any_obj_pickup_flag_currentpickup(self, self._remaining_highlight_cubes()):
            self._highlight_picks.append(self.currentpickup)
            return True
        return False

    def _highlight_subgoal_cube(self, stage):
        # Ground the actual accepted cube on its pickup frame. Otherwise the
        # first remaining cube is an oracle suggestion, not a required order.
        if len(self._highlight_picks) > stage:
            return self._highlight_picks[stage]
        remaining = self._remaining_highlight_cubes()
        return remaining[0] if remaining else None

    def _build_task_list(self):
        self._highlight_picks = []
        self.currentpickup = None
        self.timestep = 0
        self.failureflag = torch.tensor([False])
        self.successflag = torch.tensor([False])
        # Every distinct highlighted cube must be picked at least once.
        self.target_cube_pickup_counts = {name: 0 for name in self.target_cube_names}
        self.target_cube_pickup_active = {name: False for name in self.target_cube_names}

        # Define task list, each task contains a dictionary with function, name, demonstration flag, and optional failure_func
        tasks = []
        target_label = (
            getattr(self, "target_cube_color", None)
            or getattr(self, "target_cube_name", None)
            or getattr(self, "target_label", None)
            or "target"
        )
        self.target_label = target_label

        button_failure_func = lambda: is_any_obj_pickup(
            self, [cube for cube in self.all_cubes]
        )

        def wrong_cube_picked_up():
            return is_any_obj_pickup(
                self, [cube for cube in self.all_cubes if cube not in self.target_cubes]
            )

        tasks.append(
            {
                "func": lambda: is_button_pressed(self, obj=self.button),
                "name": "press the button",
                "subgoal_segment": "press the button at <>",
                "choice_label": "press button",
                "demonstration": False,
                "failure_reason": "pickup_before_button",
                "failure_func": button_failure_func,
                "solve": lambda env, planner: solve_button(
                    env, planner, obj=self.button
                ),
                "segment": self.cap_link,
            }
        )
        # Stages count completed pickups; target selection is free at every stage.
        num_targets = len(self.target_cubes)
        for cube_idx in range(num_targets):
            # If only one target cube, do not show index
            if num_targets == 1:
                task_name = "pick up the highlighted cube"
                task_subgoal = "pick up the highlighted cube at <>"
            else:
                task_name = subgoal_language.get_subgoal_with_index(
                    cube_idx, "pick up the {idx} highlighted cube"
                )
                task_subgoal = subgoal_language.get_subgoal_with_index(
                    cube_idx, "pick up the {idx} highlighted cube at <>"
                )

            tasks.append(
                {
                    "func": lambda idx=cube_idx: self._accept_highlight_pickup(idx),
                    "name": task_name,
                    "subgoal_segment": task_subgoal,
                    "choice_label": "pick up the highlighted cube",
                    "demonstration": False,
                    "failure_reason": "wrong_cube",
                    "failure_func": wrong_cube_picked_up,
                    "solve": lambda env, planner, idx=cube_idx: solve_pickup(
                        env, planner, obj=self._highlight_subgoal_cube(idx)
                    ),
                    "segment": lambda idx=cube_idx: self._highlight_subgoal_cube(idx),
                }
            )
            tasks.append(
                {
                    "func": (
                        lambda: is_obj_dropped_currentpickup(
                            self, self.target_cubes
                        )
                    ),
                    "name": f"place the cube onto the table",
                    "subgoal_segment": "place the cube onto the table",
                    "choice_label": "place the cube onto the table",
                    "demonstration": False,
                    "failure_reason": "wrong_cube",
                    "failure_func": wrong_cube_picked_up,
                    "solve": lambda env, planner: [
                        solve_putdown_whenhold(env, planner, release_z=0.01),
                    ],
                    "segment": None,
                }
            )
        tasks.append(
            {
                "func": lambda: is_button_pressed(self, obj=self.button),
                "name": "press the button",
                "subgoal_segment": "press the button at <>",
                "choice_label": "press button",
                "demonstration": False,
                "failure_reason": "wrong_cube",
                "failure_func": wrong_cube_picked_up,
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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._native_init_index = getattr(self, "_native_init_index", -1) + 1
        for cube in self.all_cubes:
            cube.set_pose(cube.initial_pose)
            cube.set_linear_velocity([0, 0, 0])
            cube.set_angular_velocity([0, 0, 0])
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos = reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)
        self._build_task_list()

        from robomme_plus.tasks.reference.utils.highlight_motion import LocalAdjustments

        self._highlight_motion = (
            LocalAdjustments(self.seed, self.cube_half_size, level=self.level,
                             start_step=self.local_adjustment_start_step)
            if self.local_adjustments else None
        )

    def _before_control_step(self):
        motion = getattr(self, "_highlight_motion", None)
        if motion is not None:
            motion.tick(self)

    def evaluate(self, solve_complete_eval=False):
        self.successflag = torch.tensor([False])
        # Keep previous failure state (once failed, always failed)
        if not hasattr(self, "failureflag") or self.failureflag is None:
            self.failureflag = torch.tensor([False])
        previous_failure = (
            bool(self.failureflag.detach().cpu().item())
            if isinstance(self.failureflag, torch.Tensor)
            else False
        )
        # If previously failed, do not reset, keep failed state; otherwise reset
        if previous_failure:
            # Keep failed state, do not reset
            pass
        else:
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

        if task_failed:
            self.failureflag = torch.tensor([True])
            logger.debug(f"Task failed: {current_task_name}")

        # If previously failed, keep failed state
        if previous_failure:
            self.failureflag = torch.tensor([True])

        ############# Rising edge detection must be placed before fail detection
        target_cubes = getattr(self, "target_cubes", [])
        target_cube_names = getattr(self, "target_cube_names", [])

        if target_cubes and not hasattr(self, "target_cube_pickup_counts"):
            self.target_cube_pickup_counts = {name: 0 for name in target_cube_names}
            self.target_cube_pickup_active = {name: False for name in target_cube_names}

        if target_cubes and not hasattr(self, "target_cube_pickup_active"):
            self.target_cube_pickup_active = {name: False for name in target_cube_names}

        # Only count when cube changes from "not picked" to "picked", avoid duplicate counting in multiple frames for same pick
        for cube, name in zip(target_cubes, target_cube_names):
            pickup_tensor = is_obj_pickup(self, cube)
            if isinstance(pickup_tensor, torch.Tensor):
                picked_now = bool(pickup_tensor.detach().cpu().any())
            else:
                picked_now = bool(pickup_tensor)

            was_picked = self.target_cube_pickup_active.get(name, False)
            if picked_now and not was_picked:
                self.target_cube_pickup_counts[name] = (
                    self.target_cube_pickup_counts.get(name, 0) + 1
                )
            self.target_cube_pickup_active[name] = picked_now

        pickup_counts = getattr(self, "target_cube_pickup_counts", {})
        counts_satisfied = len(pickup_counts) > 0 and all(
            count >= 1 for count in pickup_counts.values()
        )
        ############# Rising edge detection must be placed before fail detection

        # Success if all picked at least once (counting discrete pick events)
        if counts_satisfied and (
            all_tasks_completed or not is_newvalue_difficulty(self.difficulty)
        ):
            self.successflag = torch.tensor([True])

        # Fail if planner finished but not successful
        if all_tasks_completed and not counts_satisfied:
            from robomme_plus.scoring.failures import record_failure
            record_failure(self, "incomplete_highlights")
            self.failureflag = torch.tensor([True])
            logger.debug(f"Pickup counts not satisfied: {pickup_counts}")

        if self.failureflag.item():
            self.successflag = torch.tensor([False])
        return {
            "success": self.successflag,
            "fail": self.failureflag,
        }

    # Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):

        timestep = self.elapsed_steps
        target_cubes = getattr(self, "target_cubes", [])

        for cube, timing in zip(target_cubes, self.highlight_schedule):
            highlight_obj(
                self,
                cube,
                start_step=timing["start_step"],
                end_step=timing["end_step"],
                cur_step=timestep,
            )
        obs, reward, terminated, truncated, info = super().step(action)

        motion = getattr(self, "_highlight_motion", None)
        if motion is not None and (bool(self.failureflag.any()) or bool(self.successflag.any())):
            motion.stop(int(self.elapsed_steps.item()))

        return obs, reward, terminated, truncated, info


"""PickHighlight with three public levels mapped to the native extra tiers."""
import copy

from mani_skill.utils.registration import register_env

from robomme_plus.registry import PICK_HIGHLIGHT_LEVELS, TASKS
from robomme_plus.tasks.reference.utils.highlight_levels import (
    native_difficulty,
)


@register_env(TASKS["PickHighlight"].gym_id)
class PickHighlight(_TaskScene):
    PERCEPTUAL_VARIATIONS = True
    task_name = "PickHighlight"

    def language_goal(self):
        return (
            "first press the button, then pick up all highlighted cubes one by one, "
            "finally press the button to stop"
        )

    def __init__(
        self, *args, difficulty="hard", seed=0, _native_difficulty=None,
        highlight_mode="random", layout_version=4, local_adjustments=True,
        color_mode="similar", **kwargs
    ):
        if difficulty not in PICK_HIGHLIGHT_LEVELS:
            raise ValueError(
                f"PickHighlight difficulty must be one of {PICK_HIGHLIGHT_LEVELS}"
            )
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if kwargs.get("num_envs", 1) != 1:
            raise ValueError("PickHighlight supports one environment per instance")
        self.level = difficulty
        if type(local_adjustments) is not bool:
            raise ValueError("PickHighlight local_adjustments must be a boolean")
        self.local_adjustments = local_adjustments
        self.color_mode = validate_color_mode(color_mode)
        cube_count_range(difficulty, layout_version)
        self.layout_version = layout_version
        self.highlight_mode = resolve_highlight_mode(seed, highlight_mode)
        self.native_difficulty = native_difficulty(difficulty, seed, _native_difficulty)
        super().__init__(*args, seed=seed, difficulty=self.native_difficulty, **kwargs)


