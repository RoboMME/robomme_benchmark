from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from tests._shared.repo_paths import ensure_src_on_path, find_repo_root

_PROJECT_ROOT = find_repo_root(__file__)
ensure_src_on_path(__file__)

_SCRIPT_DIR = _PROJECT_ROOT / "scripts" / "dev"
_SCRIPT_DIR_STR = str(_SCRIPT_DIR)
if _SCRIPT_DIR_STR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR_STR)


def _load_module():
    module_path = _SCRIPT_DIR / "Env-rollout-parallel-v2.py"
    spec = importlib.util.spec_from_file_location(
        "env_rollout_parallel_v2_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeEnv:
    def __init__(self):
        self.close_called = False
        self.reset_called = False
        self.episode_success = True
        self.unwrapped = SimpleNamespace(
            failureflag=torch.tensor([False]),
            successflag=torch.tensor([True]),
            current_task_failure=False,
        )

    def reset(self):
        self.reset_called = True

    def close(self):
        self.close_called = True


def test_run_episode_forces_failure_when_bin_collision_detected(tmp_path, monkeypatch):
    module = _load_module()
    fake_env = _FakeEnv()

    monkeypatch.setattr(module, "_create_env", lambda *args, **kwargs: fake_env)
    monkeypatch.setattr(module, "_create_planner", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        module, "_wrap_planner_with_screw_then_rrt_retry", lambda planner: None
    )
    monkeypatch.setattr(module, "_execute_task_list", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        module.snapshot_utils,
        "install_snapshot_for_step",
        lambda *args, **kwargs: {
            "snapshot_enabled": True,
            "snapshot_written": True,
            "snapshot_json_path": tmp_path / "collision.json",
            "collision_detected": True,
        },
    )

    success = module._run_episode(
        env_id="VideoUnmaskSwap",
        episode=1,
        seed=1550100,
        difficulty="hard",
        output_dir=tmp_path,
    )

    assert success is False
    assert fake_env.reset_called is True
    assert fake_env.close_called is True
    assert fake_env.episode_success is False
    assert bool(fake_env.unwrapped.failureflag.item()) is True
    assert bool(fake_env.unwrapped.successflag.item()) is False
    assert fake_env.unwrapped.current_task_failure is True


def test_run_episode_keeps_success_when_no_bin_collision(tmp_path, monkeypatch):
    module = _load_module()
    fake_env = _FakeEnv()

    monkeypatch.setattr(module, "_create_env", lambda *args, **kwargs: fake_env)
    monkeypatch.setattr(module, "_create_planner", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        module, "_wrap_planner_with_screw_then_rrt_retry", lambda planner: None
    )
    monkeypatch.setattr(module, "_execute_task_list", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        module.snapshot_utils,
        "install_snapshot_for_step",
        lambda *args, **kwargs: {
            "snapshot_enabled": True,
            "snapshot_written": True,
            "snapshot_json_path": tmp_path / "no_collision.json",
            "collision_detected": False,
        },
    )

    success = module._run_episode(
        env_id="VideoUnmaskSwap",
        episode=2,
        seed=1550200,
        difficulty="hard",
        output_dir=tmp_path,
    )

    assert success is True
    assert fake_env.reset_called is True
    assert fake_env.close_called is True
    assert fake_env.episode_success is True
    assert bool(fake_env.unwrapped.failureflag.item()) is False
    assert bool(fake_env.unwrapped.successflag.item()) is True
    assert fake_env.unwrapped.current_task_failure is False
