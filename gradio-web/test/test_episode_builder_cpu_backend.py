from __future__ import annotations

import sys
import types


class _FakeEnv:
    pass


class _FakeDemonstrationWrapper:
    last_env = None
    last_kwargs = None

    def __init__(self, env, **kwargs):
        type(self).last_env = env
        type(self).last_kwargs = kwargs
        self.env = env


class _FakeFailAwareWrapper:
    last_env = None

    def __init__(self, env):
        type(self).last_env = env
        self.env = env


def test_builder_make_env_for_episode_forces_cpu_backends(monkeypatch, reload_module):
    resolver = reload_module("robomme.env_record_wrapper.episode_config_resolver")
    captured = {}

    monkeypatch.setitem(
        sys.modules,
        "robomme.env_record_wrapper.DemonstrationWrapper",
        types.SimpleNamespace(DemonstrationWrapper=_FakeDemonstrationWrapper),
    )
    monkeypatch.setitem(
        sys.modules,
        "robomme.env_record_wrapper.FailAwareWrapper",
        types.SimpleNamespace(FailAwareWrapper=_FakeFailAwareWrapper),
    )

    def fake_make(env_id, **kwargs):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return _FakeEnv()

    monkeypatch.setattr(resolver.gym, "make", fake_make)

    builder = resolver.BenchmarkEnvBuilder(
        env_id="BinFill",
        dataset="train",
        action_space="joint_angle",
        gui_render=False,
    )
    monkeypatch.setattr(builder, "resolve_episode", lambda episode_idx: (123, "hard"))

    env = builder.make_env_for_episode(7)

    assert captured["env_id"] == "BinFill"
    assert captured["kwargs"]["obs_mode"] == "rgb+depth+segmentation"
    assert captured["kwargs"]["control_mode"] == "pd_joint_pos"
    assert captured["kwargs"]["render_mode"] == "rgb_array"
    assert captured["kwargs"]["reward_mode"] == "dense"
    assert captured["kwargs"]["sim_backend"] == "physx_cpu"
    assert captured["kwargs"]["render_backend"] == "pci:0"
    assert captured["kwargs"]["seed"] == 123
    assert captured["kwargs"]["difficulty"] == "hard"
    assert _FakeDemonstrationWrapper.last_kwargs["gui_render"] is False
    assert _FakeFailAwareWrapper.last_env is env.env


def test_builder_make_env_for_episode_honors_render_backend_override(monkeypatch, reload_module):
    resolver = reload_module("robomme.env_record_wrapper.episode_config_resolver")
    captured = {}

    monkeypatch.setitem(
        sys.modules,
        "robomme.env_record_wrapper.DemonstrationWrapper",
        types.SimpleNamespace(DemonstrationWrapper=_FakeDemonstrationWrapper),
    )
    monkeypatch.setitem(
        sys.modules,
        "robomme.env_record_wrapper.FailAwareWrapper",
        types.SimpleNamespace(FailAwareWrapper=_FakeFailAwareWrapper),
    )

    def fake_make(env_id, **kwargs):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return _FakeEnv()

    monkeypatch.setattr(resolver.gym, "make", fake_make)
    monkeypatch.setenv("ROBOMME_RENDER_BACKEND", "pci:42")

    builder = resolver.BenchmarkEnvBuilder(
        env_id="BinFill",
        dataset="train",
        action_space="joint_angle",
        gui_render=False,
    )
    monkeypatch.setattr(builder, "resolve_episode", lambda episode_idx: (None, None))

    builder.make_env_for_episode(1)

    assert captured["kwargs"]["render_backend"] == "pci:42"


def test_robomme_patches_maniskill_to_preserve_pci_render_backend(reload_module):
    robomme = reload_module("robomme")
    assert robomme is not None

    from mani_skill.envs.utils.system import backend as ms_backend

    backend_name, device_id = ms_backend.parse_backend_device_id("pci:0000:00:00.0")

    assert backend_name == "pci:0000:00:00.0"
    assert device_id is None
