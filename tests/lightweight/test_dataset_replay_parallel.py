"""Lightweight checks for the synchronized replay worker protocol."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


def _load_replay_module(monkeypatch):
    class FakeBenchmarkEnvBuilder:
        pass

    robomme_module = types.ModuleType("robomme")
    wrapper_module = types.ModuleType("robomme.env_record_wrapper")
    wrapper_module.BenchmarkEnvBuilder = FakeBenchmarkEnvBuilder
    monkeypatch.setitem(sys.modules, "robomme", robomme_module)
    monkeypatch.setitem(sys.modules, "robomme.env_record_wrapper", wrapper_module)

    script_path = Path(__file__).parents[2] / "scripts" / "dataset_replay.py"
    spec = importlib.util.spec_from_file_location("dataset_replay_parallel_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Queue:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def put(self, item: dict) -> None:
        self.items.append(item)


class _Event:
    def __init__(self) -> None:
        self.wait_calls: list[int] = []

    def wait(self, timeout: int) -> bool:
        self.wait_calls.append(timeout)
        return True


def test_task_worker_waits_for_barrier_and_returns_mock_result(tmp_path, monkeypatch) -> None:
    module = _load_replay_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "_replay_task",
        lambda task_id, h5_data_dir, action_space_type, replay_number: {
            "task_id": task_id,
            "h5_path": f"{h5_data_dir}/record_dataset_{task_id}.h5",
            "episodes_requested": replay_number,
            "episodes_replayed": replay_number,
            "episodes": [],
        },
    )
    ready_queue = _Queue()
    result_queue = _Queue()
    release_event = _Event()

    module._task_worker(
        "BinFill",
        0,
        str(tmp_path / "data"),
        "joint_angle",
        10,
        str(tmp_path / "logs"),
        ready_queue,
        release_event,
        result_queue,
    )

    assert ready_queue.items == [{"task_id": "BinFill", "gpu_device": 0}]
    assert release_event.wait_calls == [module.SPAWN_START_TIMEOUT_SECONDS]
    assert result_queue.items[0]["status"] == "ok"
    assert result_queue.items[0]["episodes_replayed"] == 10
    assert Path(result_queue.items[0]["log_path"]).is_file()


def test_parallel_scheduler_declares_spawn_and_two_gpu_distribution() -> None:
    source = (Path(__file__).parents[2] / "scripts" / "dataset_replay.py").read_text(
        encoding="utf-8"
    )
    assert 'mp.get_context("spawn")' in source
    assert "GPU_DEVICE_IDS = (0, 1)" in source
    assert "release_event.wait" in source
    assert "Expected exactly 16 RoboMME tasks" in source
