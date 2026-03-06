from __future__ import annotations

import sys
import types


class _FakeDemo:
    def __init__(self):
        self.launch_kwargs = None

    def launch(self, **kwargs):
        self.launch_kwargs = kwargs
        return None


def test_main_launch_passes_ui_css(monkeypatch, reload_module):
    main = reload_module("main")
    fake_demo = _FakeDemo()
    timeout_calls = {"count": 0}

    fake_state_manager = types.SimpleNamespace(
        start_timeout_monitor=lambda: timeout_calls.__setitem__("count", timeout_calls["count"] + 1)
    )
    fake_ui_layout = types.SimpleNamespace(
        CSS="#reference_action_btn button:not(:disabled){background:#1f8b4c;}",
        create_ui_blocks=lambda: fake_demo,
    )

    monkeypatch.setitem(sys.modules, "state_manager", fake_state_manager)
    monkeypatch.setitem(sys.modules, "ui_layout", fake_ui_layout)
    monkeypatch.setenv("PORT", "7861")

    main.main()

    assert timeout_calls["count"] == 1
    assert fake_demo.launch_kwargs is not None
    assert fake_demo.launch_kwargs["server_name"] == "0.0.0.0"
    assert fake_demo.launch_kwargs["server_port"] == 7861
    assert fake_demo.launch_kwargs["css"] == fake_ui_layout.CSS
