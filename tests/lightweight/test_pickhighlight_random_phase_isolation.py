from __future__ import annotations

import importlib

import pytest
import torch

from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError

pytestmark = pytest.mark.lightweight

pickhighlight_module = importlib.import_module("robomme.robomme_env.PickHighlight")
PickHighlight = pickhighlight_module.PickHighlight


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


def _make_env(seed: int, difficulty: str = "medium") -> PickHighlight:
    env = PickHighlight.__new__(PickHighlight)
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
    return env


def _install_scene_stubs(monkeypatch, extra_scene_draws: int) -> None:
    monkeypatch.setattr(
        pickhighlight_module, "TableSceneBuilder", _FakeTableSceneBuilder
    )
    monkeypatch.setattr(
        pickhighlight_module, "is_any_obj_pickup", lambda *args, **kwargs: False
    )

    def _fake_build_button(env, center_xy, scale, generator, **kwargs):
        for _ in range(extra_scene_draws):
            torch.rand(1, generator=generator)
        env.button = _FakeActor("button")
        env.cap_link = _FakeActor("cap_link")
        return _FakeActor("button_obb")

    def _fake_spawn_random_cube(*args, color, name_prefix, generator, **kwargs):
        for _ in range(extra_scene_draws):
            torch.rand(1, generator=generator)
        return _FakeActor(name_prefix, color=color)

    monkeypatch.setattr(pickhighlight_module, "build_button", _fake_build_button)
    monkeypatch.setattr(
        pickhighlight_module, "spawn_random_cube", _fake_spawn_random_cube
    )


def _patch_randint_sequences(
    monkeypatch,
    env: PickHighlight,
    *,
    spawn_color_sequence: list[int],
    target_color_sequence: list[int] | None = None,
    target_index_sequence: list[int] | None = None,
) -> None:
    original_randint = torch.randint
    spawn_iter = iter(spawn_color_sequence)
    target_color_iter = iter(target_color_sequence or [])
    target_index_iter = iter(target_index_sequence or [])

    def _fake_randint(low, high, size, *, generator=None, **kwargs):
        if (
            generator is env.generator
            and low == 0
            and high == 3
            and tuple(size) == (1,)
        ):
            return torch.tensor([next(spawn_iter)], dtype=torch.int64)

        if generator is not None and generator is not env.generator:
            if low == 0 and high == 3 and tuple(size) == (1,):
                try:
                    return torch.tensor([next(target_color_iter)], dtype=torch.int64)
                except StopIteration:
                    pass
            if low == 0 and tuple(size) == (1,):
                try:
                    return torch.tensor([next(target_index_iter)], dtype=torch.int64)
                except StopIteration:
                    pass

        return original_randint(low, high, size, generator=generator, **kwargs)

    monkeypatch.setattr(torch, "randint", _fake_randint)


def _load_stubbed_scene(
    monkeypatch,
    *,
    seed: int,
    difficulty: str = "medium",
    extra_scene_draws: int = 0,
    spawn_color_sequence: list[int],
    target_color_sequence: list[int] | None = None,
    target_index_sequence: list[int] | None = None,
) -> PickHighlight:
    _install_scene_stubs(monkeypatch, extra_scene_draws=extra_scene_draws)
    env = _make_env(seed=seed, difficulty=difficulty)
    _patch_randint_sequences(
        monkeypatch,
        env,
        spawn_color_sequence=spawn_color_sequence,
        target_color_sequence=target_color_sequence,
        target_index_sequence=target_index_sequence,
    )
    env._load_scene({})
    return env


def test_pickhighlight_target_selection_isolated_from_scene_draws(monkeypatch) -> None:
    base_env = _load_stubbed_scene(
        monkeypatch,
        seed=504101,
        difficulty="medium",
        extra_scene_draws=0,
        spawn_color_sequence=[0, 1, 2, 0],
    )
    shifted_env = _load_stubbed_scene(
        monkeypatch,
        seed=504101,
        difficulty="medium",
        extra_scene_draws=7,
        spawn_color_sequence=[0, 1, 2, 0],
    )

    assert base_env.target_cube_names == shifted_env.target_cube_names
    assert base_env.target_cube_colors == shifted_env.target_cube_colors


def test_pickhighlight_target_selection_unique_for_multi_pick(monkeypatch) -> None:
    env = _load_stubbed_scene(
        monkeypatch,
        seed=504101,
        difficulty="medium",
        extra_scene_draws=0,
        spawn_color_sequence=[0, 0, 1, 1],
        target_color_sequence=[0, 0],
        target_index_sequence=[0, 0],
    )

    assert len(env.target_cubes) == env.configs[env.difficulty]["pickup"]
    assert len(set(env.target_cube_names)) == len(env.target_cube_names)
    assert env.target_cube_colors == ["red", "red"]


def test_pickhighlight_target_selection_varies_across_seeds(monkeypatch) -> None:
    observed = set()

    for seed in range(504101, 504106):
        env = _load_stubbed_scene(
            monkeypatch,
            seed=seed,
            difficulty="easy",
            extra_scene_draws=5,
            spawn_color_sequence=[0, 1, 2],
        )
        observed.add(tuple(env.target_cube_names))

    assert len(observed) > 1


def test_pickhighlight_missing_color_raises_scene_generation_error(
    monkeypatch,
) -> None:
    _install_scene_stubs(monkeypatch, extra_scene_draws=0)
    env = _make_env(seed=504101, difficulty="medium")
    _patch_randint_sequences(
        monkeypatch,
        env,
        spawn_color_sequence=[0, 0, 1, 1],
        target_color_sequence=[2],
    )

    with pytest.raises(SceneGenerationError, match="green"):
        env._load_scene({})
