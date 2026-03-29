from __future__ import annotations

import importlib

import pytest
import torch

pytestmark = pytest.mark.lightweight

videorepick_module = importlib.import_module("robomme.robomme_env.VideoRepick")
VideoRepick = videorepick_module.VideoRepick


class _FakeActor:
    def __init__(self, name: str, color=None):
        self.name = name
        self.color = color


class _FakeTableSceneBuilder:
    def __init__(self, env, robot_init_qpos_noise=0):
        self.env = env
        self.robot_init_qpos_noise = robot_init_qpos_noise

    def build(self) -> None:
        return None


def _make_env(seed: int, difficulty: str = "medium") -> VideoRepick:
    env = VideoRepick.__new__(VideoRepick)
    env.seed = seed
    env.difficulty = difficulty
    env.robot_init_qpos_noise = 0
    env.cube_half_size = 0.05
    env.use_demonstrationwrapper = False
    env.demonstration_record_traj = False
    env.robomme_failure_recovery = False
    env.robomme_failure_recovery_mode = None
    env.generator = torch.Generator()
    env.generator.manual_seed(seed)
    env.swap_times = 1
    env.num_repeats = 1
    env.static_flag = False
    env.start_step = 99999
    return env


def _install_scene_stubs(monkeypatch, extra_scene_draws: int) -> None:
    monkeypatch.setattr(
        videorepick_module, "TableSceneBuilder", _FakeTableSceneBuilder
    )

    def _fake_build_button(env, center_xy, scale, generator, **kwargs):
        for _ in range(extra_scene_draws):
            torch.rand(1, generator=generator)
        env.button = _FakeActor("button")
        env.button_joint = _FakeActor("button_joint")
        return _FakeActor("button_obb")

    def _fake_spawn_random_cube(*args, color, name_prefix, generator, **kwargs):
        for _ in range(extra_scene_draws):
            torch.rand(1, generator=generator)
        return _FakeActor(name_prefix, color=color)

    def _fake_rotate_points_random(region, angle_range, generator):
        for _ in range(extra_scene_draws):
            torch.rand(1, generator=generator)
        return 0, region

    monkeypatch.setattr(videorepick_module, "build_button", _fake_build_button)
    monkeypatch.setattr(
        videorepick_module, "spawn_random_cube", _fake_spawn_random_cube
    )
    monkeypatch.setattr(
        videorepick_module, "rotate_points_random", _fake_rotate_points_random
    )


def _load_stubbed_scene(
    monkeypatch, *, seed: int, difficulty: str = "medium", extra_scene_draws: int = 0
) -> VideoRepick:
    _install_scene_stubs(monkeypatch, extra_scene_draws=extra_scene_draws)
    env = _make_env(seed=seed, difficulty=difficulty)
    env._load_scene({})
    return env


def test_videorepick_num_repeats_isolated_from_scene_draws() -> None:
    env = _make_env(seed=504101, difficulty="medium")

    baseline = torch.randint(
        1,
        4,
        (1,),
        generator=env._make_generator(env._REPEAT_COUNT_SEED_OFFSET),
    ).item()
    scene_generator = torch.Generator()
    scene_generator.manual_seed(env.seed)
    for _ in range(32):
        torch.rand(1, generator=scene_generator)
    shifted = torch.randint(
        1,
        4,
        (1,),
        generator=env._make_generator(env._REPEAT_COUNT_SEED_OFFSET),
    ).item()

    assert baseline == shifted


def test_videorepick_monochrome_color_isolated_from_scene_draws(monkeypatch) -> None:
    base_env = _load_stubbed_scene(
        monkeypatch, seed=504101, difficulty="medium", extra_scene_draws=0
    )
    shifted_env = _load_stubbed_scene(
        monkeypatch, seed=504101, difficulty="medium", extra_scene_draws=7
    )

    assert len(base_env.spawned_cubes) == 3
    assert {cube.color for cube in base_env.spawned_cubes} == {
        cube.color for cube in shifted_env.spawned_cubes
    }


def test_videorepick_hard_target_selection_isolated_from_scene_draws(
    monkeypatch,
) -> None:
    base_env = _load_stubbed_scene(
        monkeypatch, seed=504101, difficulty="hard", extra_scene_draws=0
    )
    shifted_env = _load_stubbed_scene(
        monkeypatch, seed=504101, difficulty="hard", extra_scene_draws=7
    )

    assert len(base_env.spawned_cubes) == 15
    assert base_env.target_cube_1.name == shifted_env.target_cube_1.name
    assert base_env.target_cube_1.color == shifted_env.target_cube_1.color


def test_videorepick_task_randomness_can_vary_across_seeds(monkeypatch) -> None:
    observed = set()

    for seed in range(504101, 504106):
        env = _load_stubbed_scene(
            monkeypatch, seed=seed, difficulty="medium", extra_scene_draws=5
        )
        observed.add(
            (
                torch.randint(
                    1,
                    4,
                    (1,),
                    generator=env._make_generator(env._REPEAT_COUNT_SEED_OFFSET),
                ).item(),
                env.spawned_cubes[0].color,
            )
        )

    assert len(observed) > 1
