from __future__ import annotations

import sys
import types


def test_gpu_task_falls_back_to_plain_function_when_spaces_missing(monkeypatch, reload_module):
    monkeypatch.setitem(sys.modules, "spaces", None)
    runtime = reload_module("zerogpu_runtime")

    calls = []

    @runtime.gpu_task(duration=12)
    def sample(x):
        calls.append(x)
        return x + 1

    assert sample(4) == 5
    assert calls == [4]


def test_gpu_task_uses_spaces_gpu_when_available(monkeypatch, reload_module):
    recorded = {}

    def fake_gpu(*, duration, size):
        recorded["duration"] = duration
        recorded["size"] = size

        def decorator(fn):
            def wrapped(*args, **kwargs):
                recorded["called"] = True
                return fn(*args, **kwargs)

            return wrapped

        return decorator

    monkeypatch.setitem(sys.modules, "spaces", types.SimpleNamespace(GPU=fake_gpu))
    runtime = reload_module("zerogpu_runtime")

    @runtime.gpu_task(duration=33, size="xlarge")
    def sample(value):
        return value * 2

    assert sample(6) == 12
    assert recorded == {
        "duration": 33,
        "size": "xlarge",
        "called": True,
    }
