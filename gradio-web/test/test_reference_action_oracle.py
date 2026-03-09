from __future__ import annotations

import numpy as np


class _FakePose:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=np.float64)


class _FakeActor:
    def __init__(self, name: str, p):
        self.name = name
        self.pose = _FakePose(p)


class _FakeUnwrapped:
    def __init__(self, choice_label: str, current_segment=None, seg_map=None):
        self.current_choice_label = choice_label
        self.current_segment = current_segment
        self.segmentation_id_map = seg_map or {}

    def get_obs(self, unflattened=True):
        raise RuntimeError("not needed for centroid path")


class _FakeEnv:
    def __init__(self, unwrapped):
        self.unwrapped = unwrapped


def test_get_reference_action_maps_choice_and_returns_centroid_coords(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    actor = _FakeActor("cube", [0.1, 0.2, 0.3])
    unwrapped = _FakeUnwrapped(
        choice_label="pick up the cube",
        current_segment=actor,
        seg_map={7: actor},
    )
    env = _FakeEnv(unwrapped)

    monkeypatch.setattr(
        oracle_logic,
        "_build_solve_options",
        lambda env, planner, selected_target, env_id: [
            {"label": "a", "action": "pick up the cube", "available": [actor]}
        ],
    )

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    session.env = env
    session.planner = object()
    session.env_id = "BinFill"
    session.seg_raw = np.zeros((10, 10), dtype=np.int64)
    session.seg_raw[2:5, 6:9] = 7

    result = session.get_reference_action()

    assert result["ok"] is True
    assert result["option_idx"] == 0
    assert result["option_label"] == "a"
    assert result["need_coords"] is True
    assert result["coords_xy"] == [7, 3]


def test_get_reference_action_for_non_parameter_option(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    unwrapped = _FakeUnwrapped(choice_label="press the button")
    env = _FakeEnv(unwrapped)

    monkeypatch.setattr(
        oracle_logic,
        "_build_solve_options",
        lambda env, planner, selected_target, env_id: [
            {"label": "c", "action": "press the button"}
        ],
    )

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    session.env = env
    session.planner = object()
    session.env_id = "ButtonUnmask"

    result = session.get_reference_action()

    assert result["ok"] is True
    assert result["option_idx"] == 0
    assert result["need_coords"] is False
    assert result["coords_xy"] is None


def test_get_reference_action_when_choice_text_cannot_match(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    unwrapped = _FakeUnwrapped(choice_label="unknown action")
    env = _FakeEnv(unwrapped)

    monkeypatch.setattr(
        oracle_logic,
        "_build_solve_options",
        lambda env, planner, selected_target, env_id: [
            {"label": "a", "action": "pick up the cube", "available": []}
        ],
    )

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    session.env = env
    session.planner = object()
    session.env_id = "BinFill"

    result = session.get_reference_action()

    assert result["ok"] is False
    assert result["option_idx"] is None
    assert "Cannot map ground truth action" in result["message"]


def test_get_reference_action_video_place_action_still_maps_after_gradio_filter(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    target = _FakeActor("target", [0.3, 0.4, 0.5])
    unwrapped = _FakeUnwrapped(
        choice_label="drop onto",
        current_segment=target,
        seg_map={11: target},
    )
    env = _FakeEnv(unwrapped)

    monkeypatch.setattr(
        oracle_logic,
        "_build_solve_options",
        lambda env, planner, selected_target, env_id: [
            {"label": "a", "action": "pick up the cube", "available": []},
            {"label": "b", "action": "drop onto", "available": [target]},
        ],
    )

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    session.env = env
    session.planner = object()
    session.env_id = "VideoPlaceButton"
    session.seg_raw = np.zeros((12, 12), dtype=np.int64)
    session.seg_raw[5:8, 7:10] = 11

    result = session.get_reference_action()

    assert result["ok"] is True
    assert result["option_idx"] == 1
    assert result["option_label"] == "b"
    assert result["option_action"] == "drop onto"
    assert result["need_coords"] is True
    assert result["coords_xy"] == [8, 6]


def test_get_reference_action_video_place_hidden_button_fails_cleanly(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    unwrapped = _FakeUnwrapped(choice_label="press the button")
    env = _FakeEnv(unwrapped)

    monkeypatch.setattr(
        oracle_logic,
        "_build_solve_options",
        lambda env, planner, selected_target, env_id: [
            {"label": "a", "action": "pick up the cube", "available": []},
            {"label": "b", "action": "drop onto", "available": []},
        ],
    )

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    session.env = env
    session.planner = object()
    session.env_id = "VideoPlaceOrder"

    result = session.get_reference_action()

    assert result["ok"] is False
    assert result["option_idx"] is None
    assert "Cannot map ground truth action 'press the button'" in result["message"]
