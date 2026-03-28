from __future__ import annotations

import importlib

import pytest
import torch

pytestmark = pytest.mark.lightweight

pickxtimes_module = importlib.import_module("robomme.robomme_env.PickXtimes")
PickXtimes = pickxtimes_module.PickXtimes


class _FakeActor:
    def __init__(self, name: str):
        self.name = name


class _FakeTableSceneBuilder:
    def __init__(self, env, robot_init_qpos_noise=0):
        self.env = env
        self.robot_init_qpos_noise = robot_init_qpos_noise

    def build(self) -> None:
        return None


def _make_env(seed: int, difficulty: str = "medium") -> PickXtimes:
    env = PickXtimes.__new__(PickXtimes)
    env.seed = seed
    env.difficulty = difficulty
    env.robot_init_qpos_noise = 0
    env.cube_half_size = 0.05
    env.robomme_failure_recovery = False
    env.robomme_failure_recovery_mode = None
    env.use_demonstrationwrapper = False
    env.demonstration_record_traj = False
    env.num_repeats = env._sample_num_repeats()
    return env


def _install_scene_stubs(monkeypatch, extra_scene_draws: int) -> None:
    monkeypatch.setattr(
        pickxtimes_module, "TableSceneBuilder", _FakeTableSceneBuilder
    )

    def _fake_build_button(env, center_xy, scale, generator):
        env.button = _FakeActor("button")
        env.cap_link = _FakeActor("cap_link")
        return _FakeActor("button_obb")

    def _fake_spawn_random_cube(*args, name_prefix, generator, **kwargs):
        for _ in range(extra_scene_draws):
            torch.rand(1, generator=generator)
        return _FakeActor(name_prefix)

    def _fake_spawn_random_target(*args, name_prefix, generator, **kwargs):
        for _ in range(extra_scene_draws):
            torch.rand(1, generator=generator)
        return _FakeActor(name_prefix)

    monkeypatch.setattr(pickxtimes_module, "build_button", _fake_build_button)
    monkeypatch.setattr(
        pickxtimes_module, "spawn_random_cube", _fake_spawn_random_cube
    )
    monkeypatch.setattr(
        pickxtimes_module, "spawn_random_target", _fake_spawn_random_target
    )
    monkeypatch.setattr(
        pickxtimes_module, "task4recovery", lambda task_list: ([], [])
    )


def _load_stubbed_scene(
    monkeypatch, *, seed: int, difficulty: str = "medium", extra_scene_draws: int = 0
) -> PickXtimes:
    _install_scene_stubs(monkeypatch, extra_scene_draws=extra_scene_draws)
    env = _make_env(seed=seed, difficulty=difficulty)
    env._load_scene({})
    return env


def test_pickxtimes_target_selection_isolated_from_scene_draws(monkeypatch) -> None:
    base_env = _load_stubbed_scene(
        monkeypatch, seed=504101, difficulty="medium", extra_scene_draws=0
    )
    shifted_env = _load_stubbed_scene(
        monkeypatch, seed=504101, difficulty="medium", extra_scene_draws=7
    )

    assert len(base_env.all_cubes) == 3
    assert base_env.target_cube.name == shifted_env.target_cube.name
    assert base_env.target_color_name == shifted_env.target_color_name


def test_pickxtimes_num_repeats_isolated_from_scene_draws() -> None:
    env = _make_env(seed=504101, difficulty="medium")

    baseline = env._sample_num_repeats()
    scene_generator = env._make_generator()
    for _ in range(32):
        torch.rand(1, generator=scene_generator)
    shifted = env._sample_num_repeats()

    assert baseline == shifted


def test_pickxtimes_task_randomness_can_vary_across_seeds(monkeypatch) -> None:
    observed = set()

    for seed in range(504101, 504106):
        env = _load_stubbed_scene(
            monkeypatch, seed=seed, difficulty="medium", extra_scene_draws=5
        )
        observed.add((env.num_repeats, env.target_cube.name, env.target_color_name))

    assert len(observed) > 1
