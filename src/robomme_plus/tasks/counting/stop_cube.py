"""Stop Cube: scene construction, native scoring and RoboMME-Plus task configuration."""

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

import robomme_plus.control.reset as reset_panda
from robomme_plus.simulation.base import MemoryTask
from robomme_plus.simulation.difficulty import (
    is_newvalue_difficulty,
    normalize_robomme_difficulty,
)
from robomme_plus.simulation.specification import SpecRecorder
from robomme_plus.common.errors import SceneGenerationError
from robomme_plus.common.errors import (
    SceneGenerationError as _RealSceneGenerationError,
)
from robomme_plus.common.logging_utils import logger
from robomme_plus.simulation.actors import (
    build_button,
    build_purple_white_target,
    spawn_fixed_cube,
)
from robomme_plus.control.planning import (
    solve_button,
    solve_button_ready,
    solve_hold_obj_absTimestep,
)
from robomme_plus.scoring.predicates import (
    before_absTimestep,
    button_hover,
    correct_timestep,
    is_button_pressed,
    is_obj_stopped_onto,
    sequential_task_check,
)
from robomme_plus.simulation.choreography import move_straight_line


SCENE_CONFIG = {'parameters': {'interval_sample': {'sampler': 'torch.randint',
                                    'low': 27,
                                    'high_exclusive': 33,
                                    'shape': [1],
                                    'overridden_to': 30,
                                    'note': 'drawn, then immediately overridden by the constant '
                                            '30; kept so the random stream does not shift (red '
                                            'line R8)'},
                'route_rotation_deg': {'sampler': 'torch.FloatTensor(1).uniform_',
                                       'low': -30,
                                       'high': 30},
                'motion_segments': 5,
                'motion_segments_note': 'descriptive value, not read by code; actual segment '
                                        'count: 5 for the original three difficulties, max(5, '
                                        'stop_time) for xhard4',
                'steps_press_expression': 'move_interval * stop_time - move_interval / 2',
                'stop_window_expression': '[move_interval * (stop_time - 1), move_interval * '
                                          'stop_time]',
                'press_lead_steps': 'self.interval',
                'route_endpoints': {'start': [0, -0.3], 'end': [0, 0.3]},
                'recovery': 'StopCube never had failed-grasp injection; it only accepts the '
                            'recovery mode given at the entry point'},
 'positions': {'button': {'center_xy': [-0.2, 0], 'scale': 1.5, 'randomize': True},
               'target': {'xy_sampler': 'torch.FloatTensor(1).uniform_',
                          'low': -0.1,
                          'high': 0.1,
                          'z': 0.01,
                          'euler_deg': [0.0, 90.0, 0.0],
                          'radius_factor': 1.8,
                          'thickness': 0.01},
               'cube_color': {'sampler': 'torch.rand', 'shape': [3], 'alpha': 1.0},
               'cube_initial_position': [-0.3, -0.3]},
 'decision': {'move_interval_choices': [60, 80, 120],
              'stop_time_range': {'low': 2, 'high_exclusive': 6},
              'xhard1': {'move_interval_choices': [60],
                         'stop_time_range': {'low': 6, 'high_exclusive': 7}}}}


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
            "StopCube",
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
        self.stop = False

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

        self.highlight_starts = {}  # Use dictionary to store highlight start time for each button
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
            randomize=button_cfg["randomize"],
            recorder=self._spec,
            spec_path="layout.button_xy",
        )
        # avoid = [button_obb]

        target_cfg = self._sampling["positions"]["target"]
        angles = torch.deg2rad(
            torch.tensor(target_cfg["euler_deg"], dtype=torch.float32)
        )
        rotate = matrix_to_quaternion(euler_angles_to_matrix(angles, convention="XYZ"))

        target_x = (
            torch.FloatTensor(1)
            .uniform_(target_cfg["low"], target_cfg["high"], generator=generator)
            .item()
        )
        target_y = (
            torch.FloatTensor(1)
            .uniform_(target_cfg["low"], target_cfg["high"], generator=generator)
            .item()
        )
        target_x, target_y = self._spec.value("layout.target_xy", [target_x, target_y])
        self.target = build_purple_white_target(
            scene=self.scene,
            radius=self.cube_half_size * target_cfg["radius_factor"],
            thickness=target_cfg["thickness"],
            name="target",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[target_x, target_y, target_cfg["z"]], q=rotate),
        )
        color_cfg = self._sampling["positions"]["cube_color"]
        cube_color_rgb = self._spec.value(
            "objects.cube_rgb",
            torch.rand(*color_cfg["shape"], generator=generator).tolist(),
        )
        cube_color = (
            cube_color_rgb[0],
            cube_color_rgb[1],
            cube_color_rgb[2],
            color_cfg["alpha"],
        )
        self.cube = spawn_fixed_cube(
            self,
            position=[-0.3, -0.3, self.cube_half_size / 2],
            half_size=self.cube_half_size,
            color=cube_color,
            name_prefix=f"target_cube",
            yaw=0.0,  # No rotation
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos = reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)
            self.stop = False
            self.stop_timestep = None
            self._task_failed_persistent = False

            # Use generator to generate interval value, floating 5 around 20 (range 15-25)
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            interval_cfg = self._sampling["parameters"]["interval_sample"]
            # Overridden immediately; kept so the random stream does not shift, and recorded in sampling_trace (red line R8).
            self._spec.value(
                "actions.sampling_trace.interval_draw",
                torch.randint(
                    interval_cfg["low"],
                    interval_cfg["high_exclusive"],
                    tuple(interval_cfg["shape"]),
                    generator=generator,
                ).item(),
            )
            interval = interval_cfg["overridden_to"]
            self.interval = interval

            # Both branches must make the same random calls in the same order; only the interval endpoints differ.
            xhard = is_newvalue_difficulty(self.difficulty)
            decision_cfg = (
                self._sampling["decision"][self.difficulty]
                if xhard
                else self._sampling["decision"]
            )
            key_prefix = f"{self.difficulty}." if xhard else ""

            move_interval_list = list(decision_cfg["move_interval_choices"])
            idx = self._spec.value(
                "actions.move_interval_idx",
                torch.randint(
                    0, len(move_interval_list), (1,), generator=generator
                ).item(),
                decision_key=f"{key_prefix}move_interval_choices",
            )
            self.move_interval = move_interval_list[idx]

            stop_cfg = decision_cfg["stop_time_range"]
            stop_time = self._spec.value(
                "actions.stop_time",
                torch.randint(
                    stop_cfg["low"],
                    stop_cfg["high_exclusive"],
                    (1,),
                    generator=generator,
                ).item(),
                decision_key=f"{key_prefix}stop_time_range",
            )

            self.steps_press = self.move_interval * (stop_time) - self.move_interval / 2
            self.stop_time_range = (
                self.move_interval * (stop_time - 1),
                self.move_interval * (stop_time),
            )
            self.stop_time = stop_time
            # Original difficulties hardcode 5 passes in step(); xhard needs max(5, stop_time) segments so every stop pass exists.
            self.motion_segments = max(5, int(stop_time)) if xhard else 5
            self._spec.record("actions.move_interval", int(self.move_interval))
            self._spec.record("actions.motion_segments", int(self.motion_segments))
            self._spec.record("actions.steps_press", float(self.steps_press))
            self._spec.record(
                "actions.stop_window", [float(v) for v in self.stop_time_range]
            )
            # Get target xy coordinates (already randomized in _load_scene)
            target_pose = self.target.pose
            if isinstance(target_pose.p, torch.Tensor):
                target_x = target_pose.p[0, 0].item()
                target_y = target_pose.p[0, 1].item()
            else:
                target_x = target_pose.p[0]
                target_y = target_pose.p[1]
            target_center = np.array([target_x, target_y])

            # Generate random rotation angle (-30 to +30 degrees)
            rotation_cfg = self._sampling["parameters"]["route_rotation_deg"]
            rotation_angle = self._spec.value(
                "actions.rotation_deg",
                torch.FloatTensor(1)
                .uniform_(
                    rotation_cfg["low"], rotation_cfg["high"], generator=generator
                )
                .item(),
            )
            rotation_rad = np.deg2rad(rotation_angle)

            # Define original start and end coordinates (around origin (0,0))
            original_start = np.array([0, -0.3])
            original_end = np.array([0, 0.3])

            # Rotation matrix
            cos_theta = np.cos(rotation_rad)
            sin_theta = np.sin(rotation_rad)
            rotation_matrix = np.array(
                [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
            )

            # Apply rotation (around origin), then add target xy coordinates
            self.start_pos_xy = rotation_matrix @ original_start + target_center
            self.end_pos_xy = rotation_matrix @ original_end + target_center

            # Set cube initial position to rotated start point
            self.cube.set_pose(
                sapien.Pose(
                    p=[
                        self.start_pos_xy[0],
                        self.start_pos_xy[1],
                        self.cube_half_size / 2,
                    ]
                )
            )

            # Generate task list to move to each button sequentially
            tasks = []

            tasks.append(
                {
                    "func": lambda: button_hover(self, button=self.button),
                    "name": "move to the top of the button to prepare",
                    "subgoal_segment": "move to the top of the button at <> to prepare",
                    "choice_label": "move to the top of the button to prepare",
                    "demonstration": False,
                    "failure_func": None,
                    "specialflag": "swap",
                    "solve": lambda env, planner: [
                        solve_button_ready(env, planner, obj=self.button)
                    ],
                    "segment": self.cap_link,
                },
            )

            final_abs_timestep = self.steps_press - interval
            static_checkpoints = list(range(100, int(final_abs_timestep), 100))
            if not static_checkpoints or static_checkpoints[-1] != final_abs_timestep:
                static_checkpoints.append(final_abs_timestep)

            for target_timestep in static_checkpoints:
                tasks.append(
                    {
                        "func": lambda target_timestep=target_timestep: (
                            before_absTimestep(self, absTimestep=target_timestep)
                        ),
                        "name": "remain static",
                        "subgoal_segment": "remain static",
                        "choice_label": "remain static",
                        "demonstration": False,
                        "failure_func": None,
                        "specialflag": "swap",
                        "solve": lambda env, planner, target_timestep=target_timestep: (
                            solve_hold_obj_absTimestep(
                                env, planner, absTimestep=target_timestep
                            )
                        ),
                    },
                )
            tasks.append(
                {
                    "func": lambda: is_obj_stopped_onto(
                        self, obj=self.cube, target=self.target, stop=self.stop
                    ),
                    "name": "press the button to stop the cube on the target",
                    "subgoal_segment": "press the button to stop the cube on the target at <>",
                    "choice_label": "press button to stop the cube",
                    "demonstration": False,
                    "failure_func": lambda: None,
                    "solve": lambda env, planner: [
                        solve_button(env, planner, obj=self.button, without_hold=True)
                    ],
                    "segment": self.target,
                },
            )

            # Store task list for RecordWrapper use
            self.task_list = tasks

    def evaluate(self, solve_complete_eval=False):
        if not hasattr(self, "_task_failed_persistent"):
            self._task_failed_persistent = False
        self.successflag = torch.tensor([False])
        self.failureflag = (
            torch.tensor([True])
            if self._task_failed_persistent
            else torch.tensor([False])
        )

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
        task_failed = (
            task_failed or self._task_failed_persistent
        )  # Ensure overshoot is covered

        ###################################################
        if all_tasks_completed:
            correct = correct_timestep(
                self, time_range=self.stop_time_range, stop_timestep=self.stop_timestep
            )  # identify which pass pressed, stopped on, count error
            if correct != True:
                task_failed = True

        current_stop = self.stop or is_button_pressed(
            self, obj=self.button
        )  # Extra check for timing issue!
        press_before = (
            not is_obj_stopped_onto(
                self, obj=self.cube, target=self.target, stop=current_stop
            )
        ) and is_button_pressed(self, obj=self.button)
        # print(f"press_before",press_before)
        # Manually set to fail if not stopped on target
        if press_before == True:
            # import pdb; pdb.set_trace()
            task_failed = True
        ##################################################
        # Fail immediately if exceeded without press
        current_step = int(getattr(self, "elapsed_steps", 0))
        if current_step > self.move_interval * self.stop_time:
            if not all_tasks_completed:
                # The issue is that the environment continues running after the task is successfully completed,
                # eventually triggering a timeout check that incorrectly marks the episode as a failure.
                task_failed = True

        #################################################

        # If task failed, mark as failed immediately
        if task_failed:
            self._task_failed_persistent = True
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

        if is_button_pressed(
            self, obj=self.button
        ):  # Chronological issue, MUST be placed before super!!!
            self.stop = True

        obs, reward, terminated, truncated, info = super().step(action)

        # Use the rotated xy coordinates calculated in _initialize_episode
        start_pos = [
            self.start_pos_xy[0],
            self.start_pos_xy[1],
            self.cube_half_size / 2,
        ]
        end_pos = [self.end_pos_xy[0], self.end_pos_xy[1], self.cube_half_size / 2]

        # Alternate between the two waypoints so the cube makes five passes
        # Original difficulties keep range(5); xhard expands to the segment count computed in _initialize_episode.
        if is_newvalue_difficulty(getattr(self, "difficulty", None)):
            segments = range(self.motion_segments)
        else:
            segments = range(5)
        for segment in segments:
            move_straight_line(
                self,
                cube=self.cube,
                start_step=self.move_interval * segment,
                end_step=self.move_interval * (segment + 1),
                cur_step=int(self.elapsed_steps),
                start_pos=start_pos if segment % 2 == 0 else end_pos,
                end_pos=end_pos if segment % 2 == 0 else start_pos,
                stop=self.stop,
            )
        return obs, reward, terminated, truncated, info


from robomme_plus.registry import TASKS
from robomme_plus.tasks.counting.utils.environment import CountingMixin
from robomme_plus.tasks.counting.utils.variations import STOP_TARGET_SCALES


@register_env(TASKS["StopCube"].gym_id)
class StopCube(CountingMixin, _TaskScene):
    """Press the button when the moving cube reaches the target on the requested visit."""
    PERCEPTUAL_VARIATIONS = True

    task_name = "StopCube"

    def _load_scene(self, *args, **kwargs):
        # Set from the native radius each time; reconfiguration must not shrink
        # the target repeatedly. Cube geometry is independent of this scale.
        target = self._sampling["positions"]["target"]
        if not hasattr(self, "_native_target_radius_factor"):
            self._native_target_radius_factor = target["radius_factor"]
        target["radius_factor"] = self._native_target_radius_factor * STOP_TARGET_SCALES[self.level]
        return super()._load_scene(*args, **kwargs)
