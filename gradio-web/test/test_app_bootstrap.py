from __future__ import annotations

import importlib
import sys
import types


def test_app_skips_build_app_when_bootstrap_env_is_set(monkeypatch):
    calls = []

    def fake_build_app():
        calls.append("build_app")
        return object()

    monkeypatch.setenv("ROBOMME_SKIP_APP_BOOTSTRAP", "1")
    monkeypatch.setitem(
        sys.modules,
        "main",
        types.SimpleNamespace(build_app=fake_build_app, main=lambda demo=None: None),
    )
    sys.modules.pop("app", None)

    app = importlib.import_module("app")

    assert app.demo is None
    assert calls == []
