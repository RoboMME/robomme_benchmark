from __future__ import annotations

import pytest


class _FakeSession:
    def __init__(self, available=True):
        self.raw_solve_options = [{"available": available}]


def test_precheck_execute_inputs_requires_action(monkeypatch, reload_module):
    config = reload_module("config")
    callbacks = reload_module("gradio_callbacks")
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeSession(available=False))

    with pytest.raises(Exception) as excinfo:
        callbacks.precheck_execute_inputs("uid-1", None, config.UI_TEXT["coords"]["not_needed"])

    assert config.UI_TEXT["log"]["execute_missing_action"] in str(excinfo.value)


def test_precheck_execute_inputs_requires_coords_when_option_needs_it(monkeypatch, reload_module):
    config = reload_module("config")
    callbacks = reload_module("gradio_callbacks")
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeSession(available=True))

    with pytest.raises(Exception) as excinfo:
        callbacks.precheck_execute_inputs(
            "uid-1", 0, config.UI_TEXT["coords"]["select_point"]
        )

    assert config.UI_TEXT["coords"]["select_point_before_execute"] in str(excinfo.value)


def test_precheck_execute_inputs_accepts_valid_coords(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeSession(available=True))

    result = callbacks.precheck_execute_inputs("uid-1", 0, "11, 22")

    assert result is None


def test_precheck_execute_inputs_session_error(monkeypatch, reload_module):
    config = reload_module("config")
    callbacks = reload_module("gradio_callbacks")
    monkeypatch.setattr(callbacks, "get_session", lambda uid: None)

    with pytest.raises(Exception) as excinfo:
        callbacks.precheck_execute_inputs("uid-missing", 0, "1, 2")

    assert config.UI_TEXT["log"]["session_error"] in str(excinfo.value)
