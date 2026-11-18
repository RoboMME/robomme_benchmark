"""Remember the demonstrated object through a continuously moving swarm."""

import copy
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env

from robomme_plus.simulation.base import TableTask
from robomme_plus.simulation.actors import build_button
from robomme_plus.simulation.geometry import hsv_floor_rgb
from robomme_plus.common.sim import scalar, step_number
from robomme_plus.registry import TASKS
from robomme_plus.tasks.reference.utils import repick_tasks as tasks
from robomme_plus.tasks.reference.utils import repick_shapes as shapes
from robomme_plus.tasks.reference.utils.repick_controller import evaluate_tasks


@register_env(TASKS["VideoRepick"].gym_id)
class VideoRepick(TableTask):
    PERCEPTUAL_VARIATIONS = True
    task_name = "VideoRepick"

    def __init__(self, *args, difficulty="hard", seed=0, **kwargs):
        if difficulty not in tasks.LEVELS:
            raise ValueError(
                f"VideoRepick difficulty must be one of {tuple(tasks.LEVELS)}"
            )
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if kwargs.get("num_envs", 1) != 1:
            raise ValueError("VideoRepick supports one environment per instance")
        if (
            kwargs.get("native_episode_spec") is not None
            or kwargs.get("sampling_config") is not None
        ):
            raise ValueError("Replay VideoRepick with the exported seed and difficulty")
        self.level = difficulty
        self._limit_config = tasks.level_config(difficulty)
        self._limit_config["object_shape"] = shapes.shape_for_seed(seed)
        self._limit_setup = False
        self.num_repeats = 2
        super().__init__(*args, seed=seed, **kwargs)

    def _build_tracking_object(self, scene, **kwargs):
        return shapes.build_object(scene, shape=self._limit_config["object_shape"], **kwargs)

    def language_goal(self):
        repeats = {1: "once", 2: "twice"}.get(self.num_repeats, f"{self.num_repeats} times")
        return (
            "watch the video carefully, identify the object that was picked up, "
            f"then pick up and place down the same object {repeats}, "
            "finally press the button to stop"
        )

    def _load_scene(self, *args, **kwargs):
        self._limit_setup = False
        super()._load_scene(*args, **kwargs)
        # Preserve the supplied episodes' button/color stream. The two fixed
        # draws precede those cues in the recorded scene format.
        generator = torch.Generator().manual_seed(self.seed)
        torch.randint(2, 3, (1,), generator=generator)
        torch.randint(4, 5, (1,), generator=generator)
        build_button(self, center_xy=(-0.2, 0), scale=1.5, generator=generator,
                     name="button", randomize=True, randomize_range=(0.1, 0.1))
        self.button_left, self.button_joint_1 = self.button, self.button_joint
        color = [*hsv_floor_rgb(torch.rand(3, generator=generator).tolist()), 1]
        self._spec.record("objects.color_rgb", color[:3])
        count = tasks.cube_count(self.level, self.seed)
        candidates = [self._build_tracking_object(
            self.scene, initial_pose=sapien.Pose([2 + i * 0.1, 2, self.cube_half_size]),
            half_size=self.cube_half_size, color=color,
            name=f"bin_{i}" if i < 4 else f"swarm_extra_{i - 4}",
        ) for i in range(count)]
        self._repick_level_actors = candidates
        self._repick_unused_actors = []
        self._limit_diagnostics = {}
        if self._limit_config["occluder"]:
            self._limit_diagnostics["occluder"] = tasks.add_board(self)

    def _initialize_episode(self, *args, **kwargs):
        self._limit_setup = False
        super()._initialize_episode(*args, **kwargs)
        board = copy.deepcopy(self._limit_diagnostics.get("occluder"))
        self._limit_diagnostics = {
            "task": self.task_name,
            "level": self.level,
            "seed": int(self.seed),
            "config": copy.deepcopy(self._limit_config),
            "demonstration_checks": [],
        }
        if board is not None:
            self._limit_diagnostics["occluder"] = board
        self._limit_failed = False
        tasks.repick_setup(self, self._limit_config)
        for entry in self.task_list:
            if not entry["demonstration"]:
                continue
            solve = entry["solve"]

            def checked(env, planner, _entry=entry, _solve=solve):
                start = step_number(self)
                value = _solve(env, planner)
                record = {
                    "task": _entry["name"],
                    "start_step": start,
                    "end_step": step_number(self),
                    "predicate_after_solve": scalar(_entry["func"]()),
                    "planner_failed": isinstance(value, int) and value == -1,
                }
                self._limit_diagnostics["demonstration_checks"].append(record)
                if record["planner_failed"] or not record["predicate_after_solve"]:
                    raise RuntimeError(f"VideoRepick demonstration failed: {record}")
                return value

            entry["solve"] = checked

    def step(self, action):
        if self._limit_setup:
            routes = self._limit_routes
            if not getattr(self, "demonstration_record_traj", False):
                routes.active = (
                    routes.execution_start is not None
                    and step_number(self) - routes.execution_start
                    < self._limit_config["motion_tail_steps"]
                )
            routes.update()
        return BaseEnv.step(self, action)

    def on_execution_start(self):
        if self._limit_routes.execution_start is None:
            self._limit_routes.execution_start = step_number(self)
            self._limit_routes.metadata["execution_start_control_step"] = step_number(
                self
            )

    def _before_simulation_step(self):
        return BaseEnv._before_simulation_step(self)

    def _after_simulation_step(self):
        return BaseEnv._after_simulation_step(self)

    def evaluate(self, solve_complete_eval=False):
        if not self._limit_setup:
            return super().evaluate(solve_complete_eval=solve_complete_eval)
        return evaluate_tasks(self, solve_complete_eval)


