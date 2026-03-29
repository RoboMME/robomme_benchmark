from __future__ import annotations

import importlib

import pytest
import torch

pytestmark = pytest.mark.lightweight

patternlock_module = importlib.import_module("robomme.robomme_env.PatternLock")
PatternLock = patternlock_module.PatternLock


class _FakeScene:
    def __init__(self, env):
        self.env = env


class _FakeActor:
    def __init__(self, name: str, pose):
        self.name = name
        self.pose = pose


class _FakeTableSceneBuilder:
    def __init__(self, env, robot_init_qpos_noise=0):
        self.env = env
        self.robot_init_qpos_noise = robot_init_qpos_noise

    def build(self) -> None:
        return None


def _make_env(seed: int, difficulty: str = "medium") -> PatternLock:
    env = PatternLock.__new__(PatternLock)
    env.seed = seed
    env.difficulty = difficulty
    env.robot_init_qpos_noise = 0
    env.use_demonstrationwrapper = False
    env.demonstration_record_traj = False
    env.achieved_list = []
    env.match = False
    env.after_demo = False
    env.highlight_starts = {}
    env.scene = _FakeScene(env)
    return env


def _install_scene_stubs(monkeypatch, extra_scene_draws: int) -> None:
    monkeypatch.setattr(
        patternlock_module, "TableSceneBuilder", _FakeTableSceneBuilder
    )

    def _fake_build_gray_white_target(*, scene, name, initial_pose, **kwargs):
        for _ in range(extra_scene_draws):
            torch.rand(1, generator=scene.env._scene_generator)
        return _FakeActor(name, initial_pose)

    monkeypatch.setattr(
        patternlock_module, "build_gray_white_target", _fake_build_gray_white_target
    )


def _load_stubbed_scene(
    monkeypatch, *, seed: int, difficulty: str = "medium", extra_scene_draws: int = 0
) -> PatternLock:
    _install_scene_stubs(monkeypatch, extra_scene_draws=extra_scene_draws)
    env = _make_env(seed=seed, difficulty=difficulty)
    env._load_scene({})
    return env


def test_patternlock_endpoint_sampling_isolated_from_scene_draws() -> None:
    env = _make_env(seed=504101, difficulty="medium")
    num_targets = env.configs[env.difficulty]["grid"] ** 2

    baseline = torch.randperm(
        num_targets,
        generator=env._make_generator(env._PATH_NODE_SELECTION_SEED_OFFSET),
    )[:2].tolist()
    scene_generator = env._make_generator()
    for _ in range(32):
        torch.rand(1, generator=scene_generator)
    shifted = torch.randperm(
        num_targets,
        generator=env._make_generator(env._PATH_NODE_SELECTION_SEED_OFFSET),
    )[:2].tolist()

    assert baseline == shifted


def test_patternlock_path_search_isolated_from_scene_draws() -> None:
    env = _make_env(seed=504101, difficulty="medium")
    grid_size = env.configs[env.difficulty]["grid"]

    baseline, _, _, _ = patternlock_module.find_path_0_to_8(
        start=0,
        target=grid_size * grid_size - 1,
        R=grid_size,
        C=grid_size,
        diagonals=True,
        generator=env._make_generator(env._PATH_SEARCH_SEED_OFFSET),
    )
    scene_generator = env._make_generator()
    for _ in range(32):
        torch.rand(1, generator=scene_generator)
    shifted, _, _, _ = patternlock_module.find_path_0_to_8(
        start=0,
        target=grid_size * grid_size - 1,
        R=grid_size,
        C=grid_size,
        diagonals=True,
        generator=env._make_generator(env._PATH_SEARCH_SEED_OFFSET),
    )

    assert baseline == shifted


def test_patternlock_load_scene_selection_isolated_from_scene_draws(
    monkeypatch,
) -> None:
    base_env = _load_stubbed_scene(
        monkeypatch, seed=504101, difficulty="medium", extra_scene_draws=0
    )
    shifted_env = _load_stubbed_scene(
        monkeypatch, seed=504101, difficulty="medium", extra_scene_draws=7
    )

    assert [button.name for button in base_env.selected_buttons] == [
        button.name for button in shifted_env.selected_buttons
    ]


def test_patternlock_task_randomness_can_vary_across_seeds(monkeypatch) -> None:
    observed = set()

    for seed in range(504101, 504106):
        env = _load_stubbed_scene(
            monkeypatch, seed=seed, difficulty="medium", extra_scene_draws=5
        )
        observed.add(tuple(button.name for button in env.selected_buttons))

    assert len(observed) > 1
