from __future__ import annotations

import importlib

import pytest
import torch

pytestmark = pytest.mark.lightweight

routestick_module = importlib.import_module("robomme.robomme_env.RouteStick")
RouteStick = routestick_module.RouteStick


class _FakeScene:
    def __init__(self, env):
        self.env = env

    def create_actor_builder(self):
        return _FakeActorBuilder(self)


class _FakeActor:
    def __init__(self, name: str, pose):
        self.name = name
        self.pose = pose


class _FakeActorBuilder:
    def __init__(self, scene):
        self.scene = scene
        self.initial_pose = None

    def set_initial_pose(self, pose) -> None:
        self.initial_pose = pose

    def add_box_visual(self, **kwargs) -> None:
        return None

    def add_box_collision(self, **kwargs) -> None:
        return None

    def build_kinematic(self, name: str):
        for _ in range(self.scene.env._extra_scene_draws):
            torch.rand(1, generator=self.scene.env._scene_generator)
        return _FakeActor(name, self.initial_pose)


class _FakeRenderMaterial:
    def set_base_color(self, rgba) -> None:
        self.base_color = rgba


class _FakeTableSceneBuilder:
    def __init__(self, env, robot_init_qpos_noise=0):
        self.env = env
        self.robot_init_qpos_noise = robot_init_qpos_noise

    def build(self) -> None:
        return None


def _make_env(seed: int, difficulty: str = "medium") -> RouteStick:
    env = RouteStick.__new__(RouteStick)
    env.seed = seed
    env.difficulty = difficulty
    env.robot_init_qpos_noise = 0
    env.use_demonstrationwrapper = False
    env.demonstration_record_traj = False
    env.achieved_list = []
    env.match = False
    env.after_demo = False
    env.current_task_demonstration = False
    env._gripper_xy_trace = []
    env.highlight_starts = {}
    env._first_non_record_step = None
    env.z_threshold = 0.15
    env.scene = _FakeScene(env)
    env._extra_scene_draws = 0
    return env


def _install_scene_stubs(monkeypatch, extra_scene_draws: int) -> None:
    monkeypatch.setattr(
        routestick_module, "TableSceneBuilder", _FakeTableSceneBuilder
    )
    monkeypatch.setattr(
        routestick_module.sapien.render,
        "RenderMaterial",
        _FakeRenderMaterial,
    )

    def _fake_build_gray_white_target(*, scene, name, initial_pose, **kwargs):
        for _ in range(scene.env._extra_scene_draws):
            torch.rand(1, generator=scene.env._scene_generator)
        return _FakeActor(name, initial_pose)

    monkeypatch.setattr(
        routestick_module, "build_gray_white_target", _fake_build_gray_white_target
    )


def _load_stubbed_scene(
    monkeypatch, *, seed: int, difficulty: str = "medium", extra_scene_draws: int = 0
) -> RouteStick:
    _install_scene_stubs(monkeypatch, extra_scene_draws=extra_scene_draws)
    env = _make_env(seed=seed, difficulty=difficulty)
    env._extra_scene_draws = extra_scene_draws
    env._load_scene({})
    return env


def test_routestick_step_count_isolated_from_scene_draws() -> None:
    env = _make_env(seed=504101, difficulty="medium")
    length_min, length_max = env.configs[env.difficulty]["length"]

    baseline = torch.randint(
        length_min,
        length_max + 1,
        (1,),
        generator=env._make_generator(env._STEP_COUNT_SEED_OFFSET),
    ).item()
    scene_generator = env._make_generator()
    for _ in range(32):
        torch.rand(1, generator=scene_generator)
    shifted = torch.randint(
        length_min,
        length_max + 1,
        (1,),
        generator=env._make_generator(env._STEP_COUNT_SEED_OFFSET),
    ).item()

    assert baseline == shifted


def test_routestick_route_generation_isolated_from_scene_draws() -> None:
    env = _make_env(seed=504101, difficulty="medium")

    baseline = routestick_module.generate_dynamic_walk(
        [0, 2, 4, 6, 8],
        steps=4,
        allow_backtracking=env.configs[env.difficulty]["backtrack"],
        generator=env._make_generator(env._ROUTE_SELECTION_SEED_OFFSET),
    )
    scene_generator = env._make_generator()
    for _ in range(32):
        torch.rand(1, generator=scene_generator)
    shifted = routestick_module.generate_dynamic_walk(
        [0, 2, 4, 6, 8],
        steps=4,
        allow_backtracking=env.configs[env.difficulty]["backtrack"],
        generator=env._make_generator(env._ROUTE_SELECTION_SEED_OFFSET),
    )

    assert baseline == shifted


def test_routestick_swing_directions_isolated_from_scene_draws() -> None:
    env = _make_env(seed=504101, difficulty="medium")

    def _draw_directions(generator, count: int) -> list[str]:
        return [
            "clockwise"
            if torch.rand(1, generator=generator).item() < 0.5
            else "counterclockwise"
            for _ in range(count)
        ]

    baseline = _draw_directions(
        env._make_generator(env._SWING_DIRECTION_SEED_OFFSET), count=4
    )
    scene_generator = env._make_generator()
    for _ in range(32):
        torch.rand(1, generator=scene_generator)
    shifted = _draw_directions(
        env._make_generator(env._SWING_DIRECTION_SEED_OFFSET), count=4
    )

    assert baseline == shifted


def test_routestick_load_scene_task_plan_isolated_from_scene_draws(monkeypatch) -> None:
    base_env = _load_stubbed_scene(
        monkeypatch, seed=504101, difficulty="medium", extra_scene_draws=0
    )
    shifted_env = _load_stubbed_scene(
        monkeypatch, seed=504101, difficulty="medium", extra_scene_draws=7
    )

    assert [button.name for button in base_env.selected_buttons] == [
        button.name for button in shifted_env.selected_buttons
    ]
    assert base_env.swing_directions == shifted_env.swing_directions
    assert len(base_env.swing_directions) == len(base_env.selected_buttons) - 1


def test_routestick_task_randomness_can_vary_across_seeds(monkeypatch) -> None:
    observed = set()

    for seed in range(504101, 504106):
        env = _load_stubbed_scene(
            monkeypatch, seed=seed, difficulty="medium", extra_scene_draws=5
        )
        observed.add(
            (
                tuple(button.name for button in env.selected_buttons),
                tuple(env.swing_directions),
            )
        )

    assert len(observed) > 1
