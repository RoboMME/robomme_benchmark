"""Pattern Lock: scene construction, native scoring and RoboMME-Plus task configuration."""

import copy
from typing import Any, Dict, Union

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

import robomme_plus.control.reset as reset_panda
from robomme_plus.simulation.base import MemoryTask
from robomme_plus.simulation.difficulty import (
    is_newvalue_difficulty,
    normalize_robomme_difficulty,
)
from robomme_plus.simulation.specification import (
    EpisodeSpecError as _EpisodeSpecError,
)
from robomme_plus.simulation.specification import SpecRecorder
from robomme_plus.common.errors import SceneGenerationError

# NOTE: keep wildcard import for legacy helpers that the environment relies on.
# The wildcard import above shadows ``SceneGenerationError`` with a submodule; use this alias for the real class.
from robomme_plus.common.errors import (
    SceneGenerationError as _RealSceneGenerationError,
)
from robomme_plus.common.logging_utils import logger
from robomme_plus.simulation.actors import build_gray_white_target
from robomme_plus.control.planning import solve_strong_reset
from robomme_plus.scoring.predicates import (
    is_obj_swing_onto,
    reset_check,
    sequential_task_check,
)
from robomme_plus.simulation.choreography import highlight_position
from robomme_plus.tasks.imitation.utils.adjacent import find_path_0_to_8
from robomme_plus.tasks.imitation.utils.pattern_layout import (
    execution_point,
    global_path,
    grid_layouts,
    grid_subgoal,
)


SCENE_CONFIG = {'parameters': {'path_selection': {'start_end_sampler': 'torch.randperm(num_targets)[:2]',
                                   'search': 'randomized DFS allowing diagonals (find_path_0_to_8, '
                                             'diagonals=True)',
                                   'max_attempts': 1000,
                                   'exhausted_rule': 'after exhaustion use the last searched path; '
                                                     'no extra draw'},
                'path_binding': 'demo and execution reuse the same ordered grid coordinates; '
                                'the first target approach is recorded',
                'motion_template': 'for each target, solve_swingonto runs two screw motions and '
                                   'close_gripper',
                'recovery': 'reuse the fail recover mode and original generator given at entry'},
 'positions': {'grid_center': [-0.13, 0],
               'grid_center_role': 'baseline midpoint for the one-cell shift; larger execution shifts preserve the demonstration',
               'grid_spacing': 0.09,
               'execution_transform': 'seeded translation and yaw before execution; hard version 4 uses 15 cm and 10–15 degrees, otherwise one cell and 5–10 degrees; then fixed',
               'node_position_expression': 'center + (index - (n-1)/2) * spacing'},
 'decision': {'demo_duration_seconds_range': None,
              'demonstration_duration_policy': 'native',
              'grid': {'xhard1': 5},
              'path_length_range': {'xhard1': [9, 12]},
              'xhard1': {'path_search_max_attempts': 20000}}}


def _scene_gen_error(difficulty):
    """Pick the scene-generation exception class by difficulty.

    New-value (xhard) tiers return the real ``SceneGenerationError`` (a retryable task failure); the original
    three tiers return the shadowed module-level name ``SceneGenerationError`` (a submodule, so raise / except
    on it still raises TypeError; behavior unchanged).
    Usage: ``raise _scene_gen_error(self.difficulty)("message")``, ``except _scene_gen_error(self.difficulty):``;
    code that only runs on xhard paths uses ``_RealSceneGenerationError`` directly.
    """
    return (
        _RealSceneGenerationError
        if is_newvalue_difficulty(difficulty)
        else SceneGenerationError
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


    # The DFS stops at the first path inside the length range, so node counts cluster near the lower bound.


    # Combine into a dictionary

    def __init__(
        self,
        *args,
        robot_uids="panda_stick",
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
        self._spec = SpecRecorder(
            native_episode_spec,
            "PatternLock",
            {"seed": seed},
            difficulty=kwargs.get("difficulty"),
        )
        self._native_init_index = -1
        self.achieved_list = []
        self.match = False
        self.after_demo = False

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
        # self.difficulty = "hard"
        # Use seed to determine number of repetitions (1-5) arbitrarily
        generator = torch.Generator()
        generator.manual_seed(seed)

        self.highlight_starts = {}  # Use dictionary to store highlight start time for each button
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Generate 3x3 grid of buttons
        layout_cfg = self._sampling["positions"]
        decision_cfg = self._sampling["decision"]
        grid_center = list(layout_cfg["grid_center"])  # Grid center position
        grid_spacing = layout_cfg["grid_spacing"]  # Spacing between buttons

        self.buttons_grid = []
        self.button_joints_grid = []
        avoid = []
        button_index = 0

        num_rows, num_cols = 5, 8
        num_rows, num_cols = (
            decision_cfg["grid"][self.difficulty],
            decision_cfg["grid"][self.difficulty],
        )
        self._pattern_layouts = grid_layouts(
            self.seed, num_rows, grid_spacing, grid_center,
            rotate=self._imitation_config["rotate_grid"],
            execution_scale=self.execution_grid_scale,
            shift_grid=self._imitation_config.get('shift_grid', True),
            shift_distance_m=self._imitation_config.get('grid_shift_m'),
            rotation_range_deg=self._imitation_config.get('grid_rotation_range_deg', (5.0, 10.0)),
        )
        self._spec.record("scene.grid_layouts", self._pattern_layouts)
        demo_positions = self._pattern_layouts["demonstration"]["positions_xy_m"]

        for row in range(num_rows):  # 3 rows (x direction)
            for col in range(num_cols):  # 5 columns (y direction)
                x_pos, y_pos = demo_positions[button_index]

                target_name = f"target_{button_index}"

                # Create rotation quaternion for vertical target
                angles = torch.deg2rad(
                    torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32)
                )
                rotate = matrix_to_quaternion(
                    euler_angles_to_matrix(angles, convention="XYZ")
                )

                # Build purple and white target
                target = build_gray_white_target(
                    scene=self.scene,
                    radius=0.02,
                    thickness=0.01,
                    name=target_name,
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=sapien.Pose(p=[x_pos, y_pos, 0.01], q=rotate),
                )

                self.buttons_grid.append(target)
                # Note: purple_white_target doesn't have joints, so we append None
                self.button_joints_grid.append(None)
                logger.debug(
                    f"Generated target {button_index} at position ({x_pos:.3f}, {y_pos:.3f})"
                )
                button_index += 1

        self.targets_grid = self.buttons_grid

        # Generate task list to move to each button sequentially
        tasks = []

        # start_end_set = [
        #     [0, 1, 8, 9],
        #     [6, 7, 14, 15],
        #     [24, 25, 32, 33],
        #     [30, 31, 38, 39]
        # ]

        # # Randomly select 2 different sets from start_end_set
        # set_indices = torch.randperm(len(start_end_set), generator=generator)[:2].tolist()
        # start_set = start_end_set[set_indices[0]]
        # end_set = start_end_set[set_indices[1]]

        # # Randomly select one node from each set
        # start_idx = torch.randint(0, len(start_set), (1,), generator=generator).item()
        # end_idx = torch.randint(0, len(end_set), (1,), generator=generator).item()

        # num_targets = len(self.targets_grid)
        # node_choices = torch.randperm(num_targets, generator=generator)[:2]
        # start_node, end_node = node_choices.tolist()

        # path_nodes, _, _, _ = find_path_0_to_8(
        #     start=start_node,
        #     target=end_node,
        #     R=num_rows,
        #     C=num_cols,
        #     diagonals=True,
        #     generator=generator,
        # )
        # self.selected_buttons = [self.buttons_grid[i] for i in path_nodes]

        num_targets = len(self.targets_grid)
        max_attempts = self._sampling["parameters"]["path_selection"][
            "max_attempts"
        ]  # Safety limit
        max_attempts = self._xhard_decision("path_search_max_attempts")

        self._spec.identity.setdefault("difficulty", getattr(self, "difficulty", None))
        for attempt in range(max_attempts):
            if self._imitation_config["route_mode"] == "any_grid":
                path_nodes = global_path(self.seed, num_rows, self._imitation_config["path_node_count"])
                break
            # Every attempt draws as usual; failed attempts still consume random numbers.
            node_choices = torch.randperm(num_targets, generator=generator)[:2]
            start_node, end_node = node_choices.tolist()

            path_nodes, _, _, _ = find_path_0_to_8(
                start=start_node,
                target=end_node,
                R=num_rows,
                C=num_cols,
                diagonals=True,
                generator=generator,
            )

            length_range = decision_cfg["path_length_range"][self.difficulty]
            if length_range[0] <= len(path_nodes) <= length_range[1]:
                break
        else:
            raise _RealSceneGenerationError(
                f"PatternLock {self.difficulty}: {max_attempts} searches found no path whose node count is in "
                f"{decision_cfg['path_length_range'][self.difficulty]}"
            )
            # If we couldn't find a path < 5 after max_attempts, use the last one

        path_nodes = self._spec.value(
            "actions.path_nodes",
            list(path_nodes),
            decision_key=f"path_length_range.{self.difficulty}",
        )
        self._spec.record("actions.path_attempts", attempt + 1)
        self._spec.record("actions.route_mode", self._imitation_config["route_mode"])
        self._check_xhard_path(
            path_nodes,
            num_rows,
            num_cols,
            decision_cfg["path_length_range"][self.difficulty],
            adjacent=self._imitation_config["route_mode"] == "adjacent",
        )
        self.selected_buttons = [self.buttons_grid[i] for i in path_nodes]

        def endpoint_task(index, phase):
            target = self.selected_buttons[index]
            previous = path_nodes[index - 1] if index else None
            label = grid_subgoal(previous, path_nodes[index], num_cols)
            return {
                "func": lambda: self._pattern_target_confirmed(index, phase),
                "name": label,
                "subgoal_segment": label,
                "choice_label": label,
                "demonstration": phase == "demonstration",
                "failure_reason": "wrong_endpoint_stop",
                "failure_func": lambda: self._pattern_dwell.failure is not None,
                "solve": lambda env, planner: self._solve_swingonto(
                    env, planner, target=target,
                    record_swing_qpos=(phase == "demonstration" and index == 0),
                ),
            }

        tasks.extend(endpoint_task(i, "demonstration") for i in range(len(path_nodes)))

        tasks.append(
            {
                "func": lambda: reset_check(self, gripper="stick"),
                "name": "NO RECORD",
                "subgoal_segment": f"NO RECORD",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: [
                    solve_strong_reset(env, planner, gripper="stick")
                ],
            },
        )

        self.selected_buttons = [self.buttons_grid[i] for i in path_nodes]
        current_target = self.selected_buttons[0]
        tasks.append(
            {
                "func": lambda: reset_check(
                    self, gripper="stick", target_qpos=self.swing_qpos
                ),
                "name": "NO RECORD",
                "subgoal_segment": f"NO RECORD",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner, t=current_target: [
                    solve_strong_reset(
                        env, planner, gripper="stick", action=self.swing_qpos
                    )
                ],
            }
        )
        # Execution starts at home, so its first endpoint needs its own dwell.
        tasks.extend(endpoint_task(i, "execution") for i in range(len(path_nodes)))

        # Store task list for RecordWrapper use
        self.task_list = tasks

    def _xhard_decision(self, key):
        """Read an xhard-specific decision key (only called on xhard paths).

        A missing key means the given sampling_config comes from a V4 or earlier snapshot (no ``decision.xhard``);
        V4 is obsolete, so raise instead of falling back to the original three tiers' values.
        """
        xhard_cfg = self._sampling["decision"].get(self.difficulty)
        if not isinstance(xhard_cfg, dict) or key not in xhard_cfg:
            raise ValueError(
                f"PatternLock {self.difficulty}: sampling_config.decision is missing {self.difficulty}.{key}"
                " (snapshots from V4 or earlier are not usable with V5 code)"
            )
        return xhard_cfg[key]

    @staticmethod
    def _check_xhard_path(path_nodes, num_rows, num_cols, length_range, adjacent=True):
        """Check node count, uniqueness, bounds and the level's movement rule.

        In export mode the search loop already guarantees these properties, so this is a self-check; when
        replaying a frozen spec, ``SpecRecorder.value`` returns the frozen value without re-checking, and this
        guards against corrupted specs or specs from older rules. Raises ``EpisodeSpecError`` on violation.
        """
        nodes = [int(v) for v in path_nodes]
        low, high = int(length_range[0]), int(length_range[1])
        if not low <= len(nodes) <= high:
            raise _EpisodeSpecError(
                f"PatternLock xhard: path node count {len(nodes)} not in [{low}, {high}]"
            )
        if len(set(nodes)) != len(nodes):
            raise _EpisodeSpecError("PatternLock xhard: path revisits a node")
        if any(not 0 <= v < num_rows * num_cols for v in nodes):
            raise _EpisodeSpecError("PatternLock xhard: path node outside the grid")
        for a, b in zip(nodes, nodes[1:]):
            dr = abs(a // num_cols - b // num_cols)
            dc = abs(a % num_cols - b % num_cols)
            if adjacent and max(dr, dc) != 1:
                raise _EpisodeSpecError(
                    f"PatternLock xhard: path {a} → {b} is not 8-adjacent"
                )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Restore the original phase absolutely, so resets cannot accumulate a
        # translation or leave execution's targets under the demonstration.
        self._demonstration_flag = False
        self._limit_diagnostics["pattern_stops"] = {}
        self._set_pattern_layout("demonstration")
        self.timestep = 0
        self._last_task_index = None
        self._timelimit_deadlines = {}
        self.achieved_list = []
        self.match = self.after_demo = False
        self.highlight_starts = {}
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos = reset_panda.get_reset_panda_param("qpos", gripper="stick")
            self.agent.reset(qpos)

    def _set_pattern_layout(self, phase):
        from robomme_plus.simulation.choreography import highlight_obj

        layout = self._pattern_layouts[phase]
        for target, xy in zip(self.targets_grid, layout["positions_xy_m"]):
            pose = target.pose.sp
            target.set_pose(sapien.Pose([*xy, float(pose.p[2])], pose.q))
            highlight_obj(self, target, start_step=0, end_step=-1, cur_step=0)
        self.highlight_starts = {}
        for entry in getattr(self, "_position_highlight_cache", {}).values():
            entry["end_step"] = -1
            if entry.get("actor") is not None:
                entry["actor"].set_pose(sapien.Pose([0, 0, -10]))
        from robomme_plus.tasks.imitation.utils.pattern_dwell import PatternDwell

        self._pattern_grid_phase = phase
        nodes = [self.buttons_grid.index(target) for target in self.selected_buttons]
        self._pattern_dwell = PatternDwell(layout["positions_xy_m"], nodes, dt=1 / self.control_freq)
        self.achieved_list = []
        self.match = False
        self._limit_diagnostics["pattern_stops"][phase] = self._pattern_dwell.summary()
        self._limit_diagnostics["grid_layouts"] = copy.deepcopy(self._pattern_layouts)
        self._limit_diagnostics["grid_phase"] = phase

    def _prepare_execution_layout(self):
        """Move the grid before the wrapper renders its first execution frame."""
        if self._pattern_grid_phase == "execution":
            return
        start, orientation = self._trace_start_pose
        start = start.copy()
        start[:2] = execution_point(start[:2], self._pattern_layouts)
        self._trace_start_pose = (start, orientation)
        self._set_pattern_layout("execution")

    def evaluate(self, solve_complete_eval=False):
        self.successflag = torch.tensor([False])
        self.failureflag = torch.tensor([False])

        from robomme_plus.common.sim import step_number
        from robomme_plus.tasks.imitation.utils.split_stick import tip_position, drawing_allowed

        event = self._pattern_dwell.update(step_number(self), tip_position(self),
                                           drawing_allowed=drawing_allowed(self))
        if event is not None:
            self.highlight_starts[event["node"]] = event["step"]
        self.achieved_list = [self.buttons_grid[i] for i in self._pattern_dwell.accepted]
        self.match = self._pattern_grid_phase == "execution" and self._pattern_dwell.success
        self._limit_diagnostics["pattern_stops"][self._pattern_grid_phase] = self._pattern_dwell.summary()
        if self._pattern_dwell.failure is not None:
            from robomme_plus.scoring.failures import record_failure
            # Latch the specific cause before the endpoint task's generic
            # failure predicate runs (a segment deviation is not a wrong stop).
            record_failure(self, self._pattern_dwell.failure["reason"])

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

        if (
            all_tasks_completed and self.match == False
        ):  # Manually set to fail if string match fails
            logger.debug("match failure")
            from robomme_plus.scoring.failures import record_failure
            record_failure(self, "wrong_drawing_pattern")
            task_failed = True

        if self._pattern_dwell.failure is not None:
            task_failed = True

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

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # tcp_to_obj_dist = torch.linalg.norm(
        #     self.agent.tcp_pose.p - self.agent.tcp_pose.p, axis=1
        # )
        # reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        # reward = reaching_reward*0
        reward = torch.tensor([0])
        return reward

    def _pattern_target_confirmed(self, index, phase):
        return (self._pattern_grid_phase == phase
                and self._pattern_dwell.failure is None
                and len(self._pattern_dwell.accepted) > index)

    def _pattern_target_is_held(self, env, obj, target, **kwargs):
        """Refresh red only for the last confirmed node while still held."""
        node = self._pattern_dwell.held_node
        return node is not None and target is self.buttons_grid[node]

    def _update_trace_feedback(self):
        from robomme_plus.tasks.imitation.utils.tracing import update_feedback

        update_feedback(self, self._pattern_target_is_held, highlight_position)


import sys

from robomme_plus.tasks.imitation.utils import split_stick

split_stick.patch_scoring(sys.modules[__name__])
from robomme_plus.registry import TASKS
from robomme_plus.tasks.imitation.utils.environment import ImitationMixin


@register_env(TASKS["PatternLock"].gym_id)
class PatternLock(ImitationMixin, _TaskScene):
    """Confirm seeded pattern endpoints on the level's moved grid with a dwell."""
    PERCEPTUAL_VARIATIONS = True

    task_name = "PatternLock"


from robomme_plus.tasks.imitation.utils.controllers import setup_occlusion


setup_occlusion(PatternLock)
