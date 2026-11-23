"""Route Stick: scene construction, native scoring and RoboMME-Plus task configuration."""

import copy
import math
from typing import Any, Dict, Union

import numpy as np
import sapien
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
from robomme_plus.common.errors import (
    SceneGenerationError as _RealSceneGenerationError,
)
from robomme_plus.common.logging_utils import logger
from robomme_plus.simulation.actors import build_gray_white_target
from robomme_plus.control.planning import (
    solve_strong_reset,
    solve_swingonto,
    solve_swingonto_withDirection,
)
from robomme_plus.scoring.predicates import (
    is_obj_swing_onto,
    reset_check,
    sequential_task_check,
)
from robomme_plus.simulation.choreography import highlight_position
from robomme_plus.tasks.imitation.utils.route import generate_dynamic_walk


SCENE_CONFIG = {'parameters': {'configs_fallback_difficulty': 'easy',
                'walk': {'node_indices': [0, 2, 4, 6, 8],
                         'start_selection': 'randint',
                         'neighbor_order': [-1, 1],
                         'force_reverse_at_endpoint': True,
                         'direction': {'sampler': 'torch.rand',
                                       'shape': [1],
                                       'threshold': 0.5,
                                       'less_than': 'clockwise',
                                       'otherwise': 'counterclockwise'}},
                'configs': {'xhard1': {'length': [8, 10], 'backtrack': True}}},
 'positions': {'grid_center': [-0.1, 0],
               'grid_spacing_x': 0.07,
               'grid_spacing_y': 0.07,
               'rotation_center': [0, 0],
               'yaw_deg': {'scale': 60, 'subtract': 30},
               'yaw_expression': 'math.radians(u * 60 - 30)',
               'grid_spacing_x_effect': 'num_rows=1 makes the row offset always 0, so this '
                                        'parameter has no effect (dead parameter)',
               'target_builder': 'build_gray_white_target',
               'obstacle_builder': 'inlined create_actor_builder in _load_scene',
               'cylinder_radius': 0.015,
               'cylinder_height': 0.1,
               'obstacle_color': {'sampler': 'torch.rand',
                                  'shape': [3],
                                  'count': 4,
                                  'position_in_stream': 'once per obstacle cylinder, after theta '
                                                        'and before steps'},
               'walk_start': {'sampler': 'torch.randint',
                              'low': 0,
                              'high_exclusive': 5,
                              'note': 'drawn when generate_dynamic_walk is called without '
                                      'start_idx; first draw of the route random stream'},
               'tcp_trail': {'end_offset_steps': 40,
                             'disk_radius': 0.005,
                             'native_note': 'official dataset-gen d53f21a highlight_position trail '
                                            'lifetime is 40 steps'},
               'button_highlight': {'end_offset_steps': 40,
                                    'disk_radius_scale': 1.002,
                                    'native_note': 'same as official; unchanged'}},
 'decision': {'demo_duration_seconds_range': None,
              'demonstration_duration_policy': 'native',
              'xhard1': {'segment_count_range': [8, 10]}}}


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


# If direction is reversed, modify evaluate and solve


def _resolve_episode_spec(spec, task):
    """Prepare this instance's own fixed-spec copy (new-value injection); see the function of the same name in BinFill.

    With ``None`` (no ``--episode-specs`` passed) returns ``None``, and every consumer then takes the original
    random path, identical to the chain before the change.
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
        episode_spec=None,
        native_episode_spec=None,
        **kwargs,
    ):
        # Must run before any random-number call and before super().__init__()
        self._sampling = copy.deepcopy(SCENE_CONFIG)
        self._episode_spec = _resolve_episode_spec(episode_spec, "RouteStick")
        self._spec = SpecRecorder(
            native_episode_spec,
            "RouteStick",
            {"seed": seed},
            difficulty=kwargs.get("difficulty"),
        )
        self._native_init_index = -1
        self._injection_evidence = {}
        self.achieved_list = []
        self.use_demonstrationwrapper = False
        self.demonstration_record_traj = False
        self.match = False
        self.after_demo = False
        self.current_task_demonstration = False
        self._gripper_xy_trace = []
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
            self.difficulty = "easy"
            # Use seed to randomly determine number of repetitions (1-5)
        generator = torch.Generator()
        generator.manual_seed(seed)

        self.highlight_starts = {}  # Use dictionary to store highlight start time for each button
        self._first_non_record_step = None  # Start timestep for delayed highlight

        self.z_threshold = 0.15
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        camera_eye = [0.3, 0, 0.4]
        camera_target = [0, 0, -0.2]
        pose = sapien_utils.look_at(eye=camera_eye, target=camera_target)
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Generate 3x3 grid of buttons (stale; actual layout is 1 x 9)
        layout_cfg = self._sampling["positions"]
        grid_center = list(layout_cfg["grid_center"])  # Grid center position
        grid_spacing_x = layout_cfg["grid_spacing_x"]  # Spacing between buttons
        grid_spacing_y = layout_cfg["grid_spacing_y"]

        self.buttons_grid = []
        self.button_joints_grid = []
        avoid = []
        button_index = 0

        num_rows, num_cols = 1, 9
        row_center = (num_rows - 1) / 2
        col_center = (num_cols - 1) / 2

        spec = self._episode_spec
        yaw_cfg = layout_cfg["yaw_deg"]
        if spec is None:
            theta = math.radians(
                self._spec.value(
                    "layout.rotation_deg",
                    (torch.rand(1, generator=generator).item() * yaw_cfg["scale"])
                    - yaw_cfg["subtract"],
                )
            )
        else:
            theta = math.radians(float(spec["layout"]["rotation_deg"]))
        # theta=0
        for row in range(num_rows):
            for col in range(num_cols):  # Columns (y direction)
                x_pos = grid_center[0] + (row - row_center) * grid_spacing_x
                y_pos = grid_center[1] + (col - col_center) * grid_spacing_y

                orig_x, orig_y = x_pos, y_pos
                x_pos = orig_x * math.cos(theta) - orig_y * math.sin(theta)
                y_pos = orig_x * math.sin(theta) + orig_y * math.cos(theta)

                target_name = f"target_{button_index}"

                # Create rotation quaternion for vertical target
                angles = torch.deg2rad(
                    torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32)
                )
                rotate = matrix_to_quaternion(
                    euler_angles_to_matrix(angles, convention="XYZ")
                )

                # Build purple and white target
                raised_indices = {0, 2, 4, 6, 8}
                z_pos = 0.01 if button_index in raised_indices else -0.01
                target = build_gray_white_target(
                    scene=self.scene,
                    radius=0.02,
                    thickness=0.01,
                    name=target_name,
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=sapien.Pose(p=[x_pos, y_pos, z_pos], q=rotate),
                )

                self.buttons_grid.append(target)
                # Note: purple_white_target doesn't have joints, so we append None
                self.button_joints_grid.append(None)
                logger.debug(
                    f"Generated target {button_index} at position ({x_pos:.3f}, {y_pos:.3f})"
                )
                button_index += 1

        self.targets_grid = self.buttons_grid

        # Spawn white cubes on specific targets to create fixed obstacles.
        target_cube_indices = [1, 3, 5, 7]
        self.target_cube_indices = target_cube_indices
        self.target_cubes = {}
        self.cubes_on_targets = []

        for obstacle_order, target_idx in enumerate(target_cube_indices):
            if target_idx >= len(self.targets_grid):
                logger.debug(
                    f"[SwingAvoid] Skip cube spawn for target {target_idx}: index out of range."
                )
                continue

            target_actor = self.targets_grid[target_idx]
            target_pos = (
                target_actor.pose.p
                if hasattr(target_actor, "pose")
                else target_actor.get_pose().p
            )

            if isinstance(target_pos, torch.Tensor):
                target_pos = target_pos.detach().cpu().numpy()

            target_pos = np.asarray(target_pos, dtype=np.float64).reshape(-1)

            cube_position = [float(target_pos[0]), float(target_pos[1])]

            cylinder_radius = layout_cfg["cylinder_radius"]
            cylinder_height = layout_cfg["cylinder_height"]
            cylinder_half_length = cylinder_height / 2.0

            # Keep the cylinder centered so that it stands upright on the table surface.
            cylinder_angles = torch.deg2rad(
                torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32)
            )
            builder = self.scene.create_actor_builder()

            cylinder_material = sapien.render.RenderMaterial()
            if spec is None:
                random_rgb = self._spec.value(
                    f"layout.obstacle_rgb.{obstacle_order}",
                    torch.rand(3, generator=generator).tolist(),
                )
            else:
                random_rgb = [
                    float(v) for v in spec["layout"]["obstacle_rgb"][obstacle_order]
                ]
            cylinder_material.set_base_color((*random_rgb, 1))

            # Rotate upright then around its own z-axis to align with the target line.
            z_twist_mat = euler_angles_to_matrix(
                torch.tensor([0.0, 0.0, theta], dtype=torch.float32), convention="XYZ"
            )
            base_upright_mat = euler_angles_to_matrix(cylinder_angles, convention="XYZ")
            final_rot_mat = z_twist_mat @ base_upright_mat
            cylinder_quat = matrix_to_quaternion(final_rot_mat)

            builder.set_initial_pose(
                sapien.Pose(
                    p=[
                        cube_position[0],
                        cube_position[1],
                        cylinder_half_length,
                    ],
                    q=cylinder_quat.detach().cpu().numpy(),
                )
            )

            rect_length = 0.03  # 0.03
            rect_width = 0.015
            # Keep height the same as the previous cylinder; rotation keeps height along world z.
            builder.add_box_visual(
                half_size=[cylinder_half_length, rect_width, rect_length],
                material=cylinder_material,
            )
            builder.add_box_collision(
                half_size=[cylinder_half_length, rect_width, rect_length],
            )

            cube_actor = builder.build_kinematic(name=f"target_cube_{target_idx}")

            self.cubes_on_targets.append(cube_actor)
            self.target_cubes[target_idx] = cube_actor
            setattr(self, f"target_cube_{target_idx}", cube_actor)

        tasks = []

        tasks = []

        # Use the actual button actors corresponding to indices 0,2,4,6,8
        walk_cfg = self._sampling["parameters"]["walk"]
        button_indices = list(walk_cfg["node_indices"])
        self.route_button_indices = button_indices

        sampling_configs = self._sampling["parameters"]["configs"]
        fallback_difficulty = self._sampling["parameters"][
            "configs_fallback_difficulty"
        ]
        if is_newvalue_difficulty(getattr(self, "difficulty", "easy")):
            if self.difficulty not in sampling_configs:
                raise ValueError(
                    f"RouteStick {self.difficulty}: sampling_config.parameters.configs is missing the {self.difficulty} entry"
                )
            cfg = sampling_configs[self.difficulty]
        else:
            cfg = sampling_configs.get(
                getattr(self, "difficulty", "easy"),
                sampling_configs[fallback_difficulty],
            )
        length_min, length_max = cfg.get("length")
        length_decision_key = f"configs.{getattr(self, 'difficulty', 'easy')}.length"
        length_min, length_max = self._xhard_segment_count_range()
        length_decision_key = f"{self.difficulty}.segment_count_range"
        allow_backtracking = bool(cfg.get("backtrack", True))
        if spec is None:
            steps = self._spec.value(
                "objects.L",
                int(
                    torch.randint(
                        length_min, length_max + 1, (1,), generator=generator
                    ).item()
                ),
                decision_key=length_decision_key,
            )
            traj = self._spec.value(
                "actions.nodes",
                list(
                    generate_dynamic_walk(
                        button_indices,
                        steps=steps,
                        allow_backtracking=allow_backtracking,
                        generator=generator,
                        walk_config=walk_cfg,
                    )
                ),
            )
            if (
                not length_min <= int(steps) <= length_max
                or len(traj) != int(steps) + 1
            ):
                raise _EpisodeSpecError(
                    f"RouteStick {self.difficulty}: segment count {steps} / node count {len(traj)} does not match "
                    f"segment_count_range [{length_min}, {length_max}] (expected L within range and node count L+1)"
                )
        else:
            steps = int(spec["objects"]["L"])
            traj = [int(v) for v in spec["actions"]["nodes"]]
            slots = [int(v) for v in spec["actions"]["node_slots"]]
            if not length_min <= steps <= length_max:
                raise ValueError(
                    f"difficulty {self.difficulty}: spec segment count {steps} is outside [{length_min}, {length_max}]"
                )
            if len(traj) != steps + 1 or len(slots) != steps + 1:
                raise ValueError(
                    f"spec node count {len(traj)} does not match segment count {steps} (expected steps+1)"
                )
            if [button_indices[i] for i in slots] != traj:
                raise ValueError("spec node_slots and nodes are inconsistent")
            for i in range(steps):
                if abs(slots[i + 1] - slots[i]) != 1:
                    raise ValueError(
                        f"spec segment {i} is not an adjacent button pair: {slots[i]} → {slots[i + 1]}"
                    )
                if (
                    not allow_backtracking
                    and i > 0
                    and slots[i + 1] == slots[i - 1]
                    and 0 < slots[i] < len(button_indices) - 1
                ):
                    raise ValueError(
                        f"difficulty {self.difficulty} does not allow deliberate backtracking at non-endpoints: segment {i}"
                    )
            self._injection_evidence = {
                "spec_sha256": spec.get("spec_sha256"),
                "episode": spec.get("episode"),
                "rotation_deg": float(spec["layout"]["rotation_deg"]),
                "nodes": list(traj),
                "node_slots": list(slots),
            }
        self.selected_buttons = [self.buttons_grid[i] for i in traj]

        def _stick_side(actor, ref_actor=None):
            """
            Determine whether a target lies on the left or right of a reference
            target based on their y positions. When no reference is provided,
            fall back to the workspace center (y=0).
            """

            def _get_y(a):
                pos = a.pose.p if hasattr(a, "pose") else a.get_pose().p
                pos_flat = np.asarray(pos).reshape(-1)
                return pos_flat[1] if pos_flat.size >= 2 else None

            y_val = _get_y(actor)
            ref_y = _get_y(ref_actor) if ref_actor is not None else 0.0
            if y_val is None or ref_y is None:
                # Fallback to right to avoid indexing errors; should not happen.
                return "right"
            return (
                "left" if y_val > ref_y else "right"
            )  # Reverse according to robot perspective!

        # Randomly decide and record clockwise/counterclockwise direction for each solve_swingonto_withDirection
        self.swing_directions = []
        direction_cfg = walk_cfg["direction"]
        legal_directions = (direction_cfg["less_than"], direction_cfg["otherwise"])
        if spec is None:
            for order, _ in enumerate(self.selected_buttons[1:]):
                dir_flag = self._spec.value(
                    f"actions.directions.{order}",
                    direction_cfg["less_than"]
                    if torch.rand(*direction_cfg["shape"], generator=generator).item()
                    < direction_cfg["threshold"]
                    else direction_cfg["otherwise"],
                )
                self.swing_directions.append(dir_flag)
        else:
            spec_directions = list(spec["actions"]["directions"])
            if len(spec_directions) != len(self.selected_buttons) - 1:
                raise ValueError(
                    f"spec direction count {len(spec_directions)} does not match segment count {len(self.selected_buttons) - 1}"
                )
            for dir_flag in spec_directions:
                if dir_flag not in legal_directions:
                    raise ValueError(
                        f"spec contains illegal swing direction {dir_flag!r}"
                    )
                self.swing_directions.append(dir_flag)
            self._injection_evidence["directions"] = list(self.swing_directions)
        logger.debug(f"[RouteStick] swing direction list: {self.swing_directions}")

        current_target = self.selected_buttons[0]
        tasks.append(
            {
                "func": lambda t=current_target: is_obj_swing_onto(
                    self,
                    obj=self.agent.tcp,
                    target=t,
                    distance_threshold=0.03,
                    z_threshold=self.z_threshold,
                ),
                "name": "NO RECORD",
                "subgoal_segment": f"NO RECORD",
                "choice_label": "pick up the stick",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner, t=current_target: solve_swingonto(
                    env, planner, target=t, record_swing_qpos=True
                ),
            }
        )
        for i, current_target in enumerate(self.selected_buttons[1:]):
            direction = self.swing_directions[i]
            prev_target = self.selected_buttons[i]
            stick_side = _stick_side(current_target, prev_target)
            # task_name=f"rotate around the {stick_side} stick {direction}"
            task_name = f"move to the nearest {stick_side} target by circling around the board {direction}"
            tasks.append(
                {  # Decrease threshold to see if replay appears
                    "func": lambda t=current_target: is_obj_swing_onto(
                        self,
                        obj=self.agent.tcp,
                        target=t,
                        distance_threshold=0.03,
                        z_threshold=self.z_threshold,
                    ),
                    "name": task_name,
                    "subgoal_segment": task_name,
                    "choice_label": task_name,
                    "demonstration": True,
                    "failure_reason": "wrong_drawing_target",
                    "failure_func": (
                        lambda expected=current_target, last=prev_target: (
                            self._wrong_button_touch(
                                expected_button=expected, last_button=last
                            )
                        )
                    ),
                    "expected_dir": direction,
                    "solve": lambda env, planner, t=current_target, d=direction: (
                        solve_swingonto_withDirection(
                            env, planner, target=t, radius=0.2, direction=d
                        )
                    ),
                }
            )
        (
            tasks.append(
                {
                    "func": lambda: reset_check(self, gripper="stick"),
                    "name": "NO RECORD",
                    "subgoal_segment": "NO RECORD",
                    "choice_label": "place the stick into the tube",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [
                        solve_strong_reset(env, planner, timestep=200, gripper="stick")
                    ],
                },
            ),
        )

        current_target = self.selected_buttons[0]
        tasks.append(
            {
                "func": lambda: reset_check(
                    self, gripper="stick", target_qpos=self.swing_qpos
                ),
                "name": "NO RECORD",
                "subgoal_segment": f"NO RECORD",
                "choice_label": "pick up the stick",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner, t=current_target: [
                    solve_strong_reset(
                        env, planner, gripper="stick", action=self.swing_qpos
                    )
                ],
            }
        )
        from .utils.controllers import start_drawing

        tasks.append(
            {
                "func": self._accept_route_start,
                "name": "move to the starting target",
                "subgoal_segment": "move to the starting target",
                "choice_label": "move to the starting target",
                "demonstration": False,
                "failure_reason": "wrong_drawing_target",
                "failure_func": lambda: (
                    not self.demonstration_record_traj
                    and self._wrong_button_touch(self.selected_buttons[0])
                ),
                "solve": lambda env, planner: start_drawing(env.unwrapped, planner),
            }
        )
        for i, current_target in enumerate(self.selected_buttons[1:]):
            direction = self.swing_directions[i]
            prev_target = self.selected_buttons[i]
            stick_side = _stick_side(current_target, prev_target)
            # task_name=f"rotate around the {stick_side} stick {direction}"
            task_name = f"move to the nearest {stick_side} target by circling around the board {direction}"
            tasks.append(
                {
                    "func": lambda t=current_target, list=[current_target, prev_target, direction]: (
                        is_obj_swing_onto(
                            self,
                            obj=self.agent.tcp,
                            target=t,
                            distance_threshold=0.03,
                            z_threshold=self.z_threshold,
                            judge_direction_list=list,
                        )
                    ),
                    "name": task_name,
                    "subgoal_segment": task_name,
                    "choice_label": task_name,
                    "demonstration": False,
                    "failure_reason": "wrong_drawing_target",
                    "failure_func": (
                        lambda expected=current_target, last=prev_target: [
                            self._wrong_button_touch(
                                expected_button=expected, last_button=last
                            )
                        ]
                    ),
                    "expected_dir": direction,
                    "solve": lambda env, planner, t=current_target, d=direction: (
                        solve_swingonto_withDirection(
                            env, planner, target=t, radius=0.2, direction=d
                        )
                    ),
                }
            )
            # Store task list for RecordWrapper use
        self.task_list = tasks

    def _accept_route_start(self):
        # The demonstration returns to this node before execution resets to home.
        # Only a fresh execution contact can satisfy the starting subgoal.
        if self.demonstration_record_traj or self._drawing.phase != "execution":
            return False
        if split_stick.tip_position(self)[2] < 0:
            return False
        if not is_obj_swing_onto(
            self, obj=self.agent.tcp, target=self.selected_buttons[0],
            distance_threshold=0.03, z_threshold=self.z_threshold,
        ):
            return False
        self._gripper_xy_trace = []  # The approach must not determine the first curve's direction.
        return True

    def _xhard_segment_count_range(self):
        """Segment count range L for xhard, read from ``decision.xhard.segment_count_range``.

        When sampling it equals the default of ``config_xhard4.length``; on replay the sampling_config comes
        from the spec header, so the frozen value is read. A missing key means a V4 or earlier snapshot was
        passed (V4 headers do not freeze the L range); V4 is obsolete, so this raises instead of falling back
        to class attributes.
        """
        xhard_cfg = self._sampling["decision"].get(self.difficulty)
        value = (
            xhard_cfg.get("segment_count_range")
            if isinstance(xhard_cfg, dict)
            else None
        )
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
            or not 1 <= value[0] <= value[1]
        ):
            raise ValueError(
                f"RouteStick {self.difficulty}: sampling_config.decision.{self.difficulty}.segment_count_range is missing or not "
                f"a [lower, upper] pair of positive integers (got {value!r}; V4 and earlier snapshots are unusable on V5 code)"
            )
        return int(value[0]), int(value[1])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos = reset_panda.get_reset_panda_param("qpos", gripper="stick")
            self.agent.reset(qpos)
            self.failureflag = torch.tensor([False])

    def evaluate(self, solve_complete_eval=False):
        previous_failure = getattr(self, "failureflag", torch.tensor([False]))
        if isinstance(previous_failure, torch.Tensor):
            failure_latched = bool(previous_failure.detach().cpu().item())
        else:
            failure_latched = bool(previous_failure)
        self.successflag = torch.tensor([False])
        self.failureflag = (
            torch.tensor([True]) if failure_latched else torch.tensor([False])
        )
        had_latched_fail = failure_latched

        # Direction follows white drawing points at the real tip. A raised
        # excursion must neither fix a wrong curve nor invalidate a correct one.
        # Repeated evaluation of one control step must not reweight the trace.
        if self.current_task_demonstration == False and split_stick.drawing_allowed(self):
            xy = split_stick.tip_position(self)[:2]
            if hasattr(self, "_route_to_rest"):
                xy = self._route_to_rest(xy)
            gripper_xy = torch.as_tensor(xy).clone()
            step = int(self.elapsed_steps.item())
            if not self._gripper_xy_trace or int(self._gripper_xy_trace[-1][0]) != step:
                self._gripper_xy_trace.append((step, gripper_xy))

        # Use encapsulated sequence task check function
        if (
            self.use_demonstrationwrapper == False
        ):  # change subgoal after planner ends during recording
            if solve_complete_eval == True:
                allow_subgoal_change_this_timestep = True
            else:
                allow_subgoal_change_this_timestep = False
        else:  # during demonstration, video needs to call evaluate(solve_complete_eval) video ends and flag changes in demonstrationwrapper
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

        # If task failed, mark as failed immediately, and keep fail thereafter
        if task_failed:
            self.failureflag = torch.tensor([True])
            if not had_latched_fail:
                logger.debug(f"Task failed: {current_task_name}")

        # If static_check succeeds or all tasks completed, set success flag
        if all_tasks_completed and not task_failed and not bool(self.failureflag.item()):
            self.successflag = torch.tensor([True])

        # # Check if a swing task has just been completed
        # if after_demo_active:
        #     # If current_task_name jumps, record adjacent two targets in order
        #     last_logged_name = getattr(self, "_last_logged_task_name", None)
        #     if current_task_name != last_logged_name:
        #         change_idx = getattr(self, "_swing_task_change_idx", 0)
        #         buttons = getattr(self, "selected_buttons", [])
        #         if change_idx + 1 < len(buttons):
        #             first_target = buttons[change_idx]
        #             second_target = buttons[change_idx + 1]
        #             expected_dir = (
        #                 self.swing_directions[change_idx]
        #                 if change_idx < len(getattr(self, "swing_directions", []))
        #                 else None
        #             )
        #             self._swing_success_history = [
        #                 {"step": cur_step, "target": first_target, "expected_dir": None},
        #                 {"step": cur_step, "target": second_target, "expected_dir": expected_dir},
        #             ]
        #             self._swing_task_change_idx = change_idx + 1
        #         self._last_logged_task_name = current_task_name

        #     # When there are two recent successful swings (different targets), judge gripper trajectory on left/right side of line between two targets and print
        #     if len(self._swing_success_history) == 2:
        #         first, second = self._swing_success_history
        #         pair_key = (first["step"], second["step"])
        #         if first["target"] is not second["target"] and self._last_swing_pair_reported_step != pair_key:
        #             start, end = first["step"], second["step"]
        #             segment = [(s, xy) for s, xy in self._gripper_xy_trace if start <= s <= end]
        #             # Only keep trajectory from first timestamp, avoid list growing infinitely
        #             self._gripper_xy_trace = [(s, xy) for s, xy in self._gripper_xy_trace if s >= start]

        #             t1 = torch.as_tensor(first["target"].pose.p[0][:2]).detach().cpu()
        #             t2 = torch.as_tensor(second["target"].pose.p[0][:2]).detach().cpu()
        #             line_vec = t2 - t1
        #             cross_vals = []
        #             for _, xy in segment:
        #                 rel = xy - t1
        #                 cross_vals.append(float(line_vec[0] * rel[1] - line_vec[1] * rel[0]))

        #             if cross_vals:
        #                 avg_cross = sum(cross_vals) / len(cross_vals)
        #                 # According to current coordinate system, positive cross product direction should be considered clockwise
        #                 side = "clockwise" if avg_cross > 0 else "counterclockwise" if avg_cross <0 else "on the line"
        #                 expected_dir = second.get("expected_dir")
        #                 if expected_dir and side != "on the line" and side != expected_dir:
        #                     print("direction mistake!!!")
        #                 print(f"Gripper path from step {start} to {end} stayed on the {side} side of the directed line between the last two targets.")
        #             self._last_swing_pair_reported_step = pair_key

        return {
            "success": self.successflag,
            "fail": self.failureflag,
        }

    def direction_fail(self, judge_direction_list=None):
        from robomme_plus.scoring.failures import record_failure

        if judge_direction_list is None:
            return True

        # judge_direction_list format is [current_target, prev_target, expected_dir]
        try:
            current_target, prev_target, expected_dir = judge_direction_list
        except Exception:
            # Unable to judge direction when parameter is abnormal, keep unfinished status
            record_failure(self, "failure_check_error")
            self.failureflag = torch.tensor([True])
            return False

        trace = getattr(self, "_gripper_xy_trace", [])

        # Calculate vector from previous target to current target
        prev_xy = torch.as_tensor(prev_target.pose.p[0][:2]).detach().cpu()
        curr_xy = torch.as_tensor(current_target.pose.p[0][:2]).detach().cpu()
        if hasattr(self, "_route_to_rest"):
            prev_xy = torch.as_tensor(self._route_to_rest(prev_xy))
            curr_xy = torch.as_tensor(self._route_to_rest(curr_xy))
        line_vec = curr_xy - prev_xy

        # If two targets coincide or no trajectory, unable to judge direction
        if torch.norm(line_vec) < 1e-6 or len(trace) == 0:
            record_failure(self, "invalid_route_segment")
            self.failureflag = torch.tensor([True])
            return False

        cross_vals = []
        for _, xy in trace:
            xy = torch.as_tensor(xy).detach().cpu()
            rel = xy - prev_xy
            cross_vals.append(float(line_vec[0] * rel[1] - line_vec[1] * rel[0]))

        # Clear trajectory immediately after consumption, convenient for next segment judgment
        self._gripper_xy_trace = []

        if len(cross_vals) == 0:
            record_failure(self, "invalid_route_segment")
            self.failureflag = torch.tensor([True])
            return False

        avg_cross = sum(cross_vals) / len(cross_vals)
        # Positive cross product in current coordinate system is considered clockwise
        side = (
            "clockwise"
            if avg_cross > 0
            else "counterclockwise"
            if avg_cross < 0
            else "on the line"
        )

        # Unable to judge direction along line; need to retry
        if side == "on the line":
            record_failure(self, "invalid_route_segment")
            self.failureflag = torch.tensor([True])
            return False

        expected_dir = str(expected_dir).lower()
        if side != expected_dir:
            record_failure(self, "wrong_route_direction")
            logger.debug(f"direction mistake: expected {expected_dir}, got {side}")
            self.failureflag = torch.tensor([True])
            return False

        return True

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        reward = torch.tensor([0])
        return reward

    def _update_trace_feedback(self):
        from robomme_plus.tasks.imitation.utils.tracing import update_feedback

        update_feedback(self, is_obj_swing_onto, highlight_position)

    def _wrong_button_touch(self, expected_button, last_button=None):
        # Only visible route nodes are drawing targets. The interleaved hidden
        # markers locate boards; touching their positions is not a wrong node.
        for index in self.route_button_indices:
            button = self.buttons_grid[index]
            if button is expected_button:
                continue
            if last_button is not None and button is last_button:
                continue
            if is_obj_swing_onto(self, obj=self.agent.tcp, target=button):
                return True
        return False


import sys

from robomme_plus.tasks.imitation.utils import split_stick

split_stick.patch_scoring(sys.modules[__name__])
from robomme_plus.registry import TASKS
from robomme_plus.tasks.imitation.utils.environment import ImitationMixin


@register_env(TASKS["RouteStick"].gym_id)
class RouteStick(ImitationMixin, _TaskScene):
    """Repeat a visible seeded route; medium/hard shorten the execution stick."""
    PERCEPTUAL_VARIATIONS = True

    task_name = "RouteStick"

    def _load_scene(self, options):
        self._route_motion_parameters = None
        layout = self.execution_route_layout
        if layout is not None and layout["version"] >= 2:
            # Fixed-height layouts keep the selected height throughout both phases.
            self._sampling["positions"]["cylinder_height"] = .1 * layout["board_height_scale"]
        super()._load_scene(options)
        if self.execution_route_layout is None:
            return
        if layout["version"] >= 2:
            # Raw joint reset can sweep the long stick through a taller board.
            # Lift clear before the hidden return-home part of the demonstration.
            reset_home = [entry for entry in self.task_list if entry["demonstration"]][-2]
            original_reset = reset_home["solve"]

            def lifted_reset(env, planner):
                from .utils.controllers import move_pose, rotation
                from robomme_plus.common.sim import position

                point = position(self.agent.tcp, False).copy()
                point[2] = max(point[2], .1 * layout["board_height_scale"] + .06)
                move_pose(planner, point, rotation(self.agent.tcp))
                return original_reset(env, planner)

            reset_home["solve"] = lifted_reset
        from .utils.route_layout import execution_points

        self._route_demo_target_poses = [target.pose.sp for target in self.targets_grid]
        self._route_demo_boards = list(self.cubes_on_targets)
        self._route_demo_board_poses = [board.pose.sp for board in self._route_demo_boards]
        xy = np.array([pose.p[:2] for pose in self._route_demo_target_poses])
        self._route_demo_center = xy.mean(axis=0)
        layout = self.execution_route_layout
        self._route_layouts = {
            "parameters": copy.deepcopy(layout),
            "demonstration": {"positions_xy_m": xy.tolist(),
                              "board_height_m": self._sampling["positions"]["cylinder_height"]},
            "execution": {"positions_xy_m": execution_points(xy, self._route_demo_center, layout).tolist(),
                          "board_height_m": .1 * layout["board_height_scale"]},
        }
        self._route_execution_boards = []
        colors = self._spec.to_dict()["layout"]["obstacle_rgb"]
        for index, source in enumerate(self._route_demo_boards):
            half = np.asarray(source._bodies[0].get_collision_shapes()[0].half_size).copy()
            half[0] = .05 * layout["board_height_scale"]  # native box local x is world vertical
            color = colors.get(str(index), colors.get(index))
            material = sapien.render.RenderMaterial(base_color=[*color, 1], roughness=.8)
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(half_size=half, material=material)
            builder.add_box_collision(half_size=half)
            builder.initial_pose = sapien.Pose([0, 0, -10])
            self._route_execution_boards.append(builder.build_kinematic(name=f"execution_route_board_{index}"))
        if not layout.get("board_collision", True):
            from .utils.route_layout import disable_board_collisions
            disable_board_collisions(self._route_demo_boards + self._route_execution_boards)
        self._route_demo_depth_m = self._imitation_config["route_depth_m"]
        self._route_layout_phase = "demonstration"
        self._imitation_config["execution_route_layout"] = copy.deepcopy(layout)
        self._route_motion_parameters = copy.deepcopy(layout.get("board_motion"))

    def _set_route_layout(self, phase):
        from .utils.controllers import turned
        from .utils.route_layout import execution_points

        execution = phase == "execution"
        layout = self.execution_route_layout
        points = self._route_layouts[phase]["positions_xy_m"]
        for target, pose, xy in zip(self.targets_grid, self._route_demo_target_poses, points):
            target.set_pose(sapien.Pose([*xy, float(pose.p[2])], pose.q))
        for source, changed, pose in zip(self._route_demo_boards, self._route_execution_boards,
                                         self._route_demo_board_poses):
            source.set_pose(sapien.Pose([0, 0, -10]) if execution else pose)
            if execution:
                xy = execution_points(pose.p[:2], self._route_demo_center, layout)
                q = matrix_to_quaternion(torch.tensor(turned(
                    sapien.Pose(q=pose.q).to_transformation_matrix()[:3, :3], math.radians(layout["yaw_deg"])
                ), dtype=torch.float32)).numpy()
                changed.set_pose(sapien.Pose([*xy, .05 * layout["board_height_scale"]], q))
            else:
                changed.set_pose(sapien.Pose([0, 0, -10]))
        self.cubes_on_targets = self._route_execution_boards if execution else self._route_demo_boards
        self.target_cubes = dict(zip(self.target_cube_indices, self.cubes_on_targets))
        for index, board in self.target_cubes.items():
            setattr(self, f"target_cube_{index}", board)
        self._imitation_config["route_depth_m"] = self._route_demo_depth_m * (layout["spacing_scale"] if execution else 1)
        self._route_layout_phase = phase
        self._route_motion_rest_poses = [board.pose.sp for board in self.cubes_on_targets]
        self._route_motion_target_rest_poses = [target.pose.sp for target in self.targets_grid]
        self._route_motion_angle = 0.
        self._route_motion_physics_steps = 0
        self._route_departure = None
        if self._route_motion_parameters is not None and self._route_motion_parameters["mode"] in ("arrival_boards", "departure_boards"):
            from .utils.route_departure import DepartureCrossings
            self._route_departure = DepartureCrossings(self._route_motion_parameters)
        self.highlight_starts = {}
        for entry in getattr(self, "_position_highlight_cache", {}).values():
            entry["end_step"] = -1
            if entry.get("actor") is not None:
                entry["actor"].set_pose(sapien.Pose([0, 0, -10]))
        if hasattr(self, "_limit_diagnostics"):
            if phase == "demonstration":
                self._limit_diagnostics.pop("crossing_demonstration_waits", None)
                self._limit_diagnostics.pop("departure_demonstration_responses", None)
            self._limit_diagnostics["route_layouts"] = copy.deepcopy(self._route_layouts)
            self._limit_diagnostics["route_layout_phase"] = phase
            if self._route_motion_parameters is not None:
                self._limit_diagnostics["route_board_motion"] = dict(
                    parameters=copy.deepcopy(self._route_motion_parameters), phase=phase,
                    time_s=0., angles_deg=[0.] * len(self.cubes_on_targets),
                    centers_xy_m=[pose.p[:2].tolist() for pose in self._route_motion_rest_poses])

    def _route_to_rest(self, xy):
        """Express a measured point in this phase's non-oscillating layout frame."""
        motion = getattr(self, "_route_motion_parameters", None)
        if motion is None or motion["mode"] != "rigid_layout":
            return np.asarray(xy, float)
        from .utils.route_motion import rotate_points
        pivot = self._route_motion_target_rest_poses[motion["pivot_target_index"]].p[:2]
        return rotate_points(xy, pivot, -self._route_motion_angle)

    def _before_simulation_step(self):
        super()._before_simulation_step()
        motion = getattr(self, "_route_motion_parameters", None)
        if motion is None or not hasattr(self, "_route_motion_rest_poses"):
            return
        phase = self._route_layout_phase
        if phase not in motion["phases"]:
            return
        if phase == "demonstration":
            if not self.demonstration_record_traj:
                return
        elif getattr(getattr(self, "_drawing", None), "phase", None) != "execution":
            return
        from transforms3d.quaternions import qmult
        from .utils.route_motion import board_swing, layout_swing
        from robomme_plus.common.sim import position

        self._route_motion_physics_steps += 1
        seconds = self._route_motion_physics_steps / self.sim_freq
        poses = self._route_motion_rest_poses
        actors = list(self.cubes_on_targets)
        crossing = None
        if motion["mode"] in ("arrival_boards", "departure_boards"):
            from .utils.route_departure import interception_contact
            curve = getattr(self, "_route_curve_tasks", {}).get(phase, {}).get(int(self.timestep))
            if curve is not None:
                start, end = self.selected_buttons[curve:curve+2]
                indices = {id(target):i for i,target in enumerate(self.targets_grid)}
                tip, touching = interception_contact(self)
                self._route_departure.update(curve=curve,start_index=indices[id(start)],end_index=indices[id(end)],
                    direction=self.swing_directions[curve],tip_xy=tip[:2],
                    start_xy=position(start,False)[:2],drawing=touching,time_s=seconds)
            target_poses = self._route_motion_target_rest_poses
            row = target_poses[-1].p[:2]-target_poses[0].p[:2]
            scale = self.execution_route_layout["spacing_scale"] if phase == "execution" else 1.
            centers,crossing = self._route_departure.positions([p.p[:2] for p in poses],row,seconds,scale)
            angles = np.zeros(len(poses))
        elif motion["mode"] == "crossing_boards":
            from .utils.route_crossing import crossing_positions
            target_poses = self._route_motion_target_rest_poses
            row = target_poses[-1].p[:2] - target_poses[0].p[:2]
            scale = self.execution_route_layout["spacing_scale"] if phase == "execution" else 1.
            centers, crossing = crossing_positions([p.p[:2] for p in poses], row, motion, seconds, scale)
            angles = np.zeros(len(poses))
        elif motion["mode"] == "rigid_layout":
            poses = poses + self._route_motion_target_rest_poses
            actors += list(self.targets_grid)
            pivot = self._route_motion_target_rest_poses[motion["pivot_target_index"]].p[:2]
            centers, angle = layout_swing([p.p[:2] for p in poses], pivot, motion, seconds)
            self._route_motion_angle = angle
            angles = np.full(len(poses), angle)
        else:
            pivots = np.array([position(self.targets_grid[i], False)[:2]
                               for i in motion["pivot_target_indices"]])
            centers, angles = board_swing([p.p[:2] for p in poses], pivots, motion, seconds)
        for board, rest, xy, angle in zip(actors, poses, centers, angles):
            q = qmult([math.cos(angle / 2), 0., 0., math.sin(angle / 2)], rest.q)
            target = sapien.Pose([*xy, float(rest.p[2])], q)
            for body in board._bodies:
                # Kinematic targets generate contact velocities during physics;
                # moving the visual pose alone would not push the robot correctly.
                body.set_kinematic_target(target)
        self._limit_diagnostics["route_board_motion"] = dict(
            parameters=copy.deepcopy(motion), phase=phase, time_s=seconds,
            angles_deg=np.degrees(angles[:len(self.cubes_on_targets)]).tolist(),
            centers_xy_m=centers[:len(self.cubes_on_targets)].tolist(),
            target_centers_xy_m=centers[len(self.cubes_on_targets):].tolist()
            if motion["mode"] == "rigid_layout" else [position(t, False)[:2].tolist() for t in self.targets_grid])
        if crossing is not None:
            self._limit_diagnostics["route_board_motion"]["crossing_states"] = crossing

    def _initialize_episode(self, env_idx, options):
        if self.execution_route_layout is not None:
            self.__dict__["_demonstration_flag"] = False
            self._set_route_layout("demonstration")
            self.timestep = 0
            self._gripper_xy_trace = []
        super()._initialize_episode(env_idx, options)

    def _prepare_execution_layout(self):
        if self.execution_route_layout is None or self._route_layout_phase == "execution":
            return
        from .utils.controllers import turned
        from .utils.route_layout import execution_points

        point, orientation = self._trace_start_pose
        point = point.copy()
        point[:2] = execution_points(point[:2], self._route_demo_center, self.execution_route_layout)
        self._trace_start_pose = (point, turned(orientation, math.radians(self.execution_route_layout["yaw_deg"])))
        self._set_route_layout("execution")


from robomme_plus.tasks.imitation.utils.controllers import setup_occlusion


setup_occlusion(RouteStick)
