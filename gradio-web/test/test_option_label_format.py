from __future__ import annotations

import numpy as np


class _FakeUnwrapped:
    def __init__(self):
        self.segmentation_id_map = {}


class _FakeEnv:
    def __init__(self):
        self.unwrapped = _FakeUnwrapped()
        self.frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
        self.wrist_frames = []


class _FakeObsWrapperEnv:
    def __init__(self, front_rgb_list, wrist_rgb_list):
        self.unwrapped = _FakeUnwrapped()
        self._last_obs = {
            "front_rgb_list": front_rgb_list,
            "wrist_rgb_list": wrist_rgb_list,
        }



def test_available_options_use_label_plus_action(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    monkeypatch.setattr(
        oracle_logic,
        "_fetch_segmentation",
        lambda env: np.zeros((1, 8, 8), dtype=np.int64),
    )
    monkeypatch.setattr(
        oracle_logic,
        "_build_solve_options",
        lambda env, planner, selected_target, env_id: [
            {"label": "a", "action": "pick up the cube", "available": [1]},
            {"label": "b", "action": "put it down", "available": []},
        ],
    )

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    session.env = _FakeEnv()
    session.planner = object()
    session.env_id = "BinFill"
    session.color_map = {}

    _img, msg = session.update_observation()

    assert msg == "Ready"
    assert session.available_options == [
        ("a. pick up the cube", 0),
        ("b. put it down", 1),
    ]
    assert session.raw_solve_options[0]["label"] == "a"


def test_build_solve_options_filters_press_button_for_video_place_envs(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    base_options = [
        {"label": "a", "action": "pick up the cube", "available": [1]},
        {"label": "b", "action": "drop onto", "available": [2]},
        {"label": "c", "action": "press the button"},
    ]
    monkeypatch.setattr(
        oracle_logic,
        "get_vqa_options",
        lambda env, planner, selected_target, env_id: list(base_options),
    )

    filtered_button = oracle_logic._build_solve_options(None, None, {}, "VideoPlaceButton")
    filtered_order = oracle_logic._build_solve_options(None, None, {}, "VideoPlaceOrder")
    unfiltered_other = oracle_logic._build_solve_options(None, None, {}, "ButtonUnmask")

    assert [opt["label"] for opt in filtered_button] == ["a", "b"]
    assert [opt["action"] for opt in filtered_button] == ["pick up the cube", "drop onto"]
    assert [opt["label"] for opt in filtered_order] == ["a", "b"]
    assert [opt["action"] for opt in unfiltered_other] == [
        "pick up the cube",
        "drop onto",
        "press the button",
    ]


def test_update_observation_no_seg_vis_base_fallback(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    seg_vis = np.zeros((6, 6, 3), dtype=np.uint8)
    seg_vis[:, :, 0] = 10  # B
    seg_vis[:, :, 1] = 20  # G
    seg_vis[:, :, 2] = 30  # R

    monkeypatch.setattr(
        oracle_logic,
        "_fetch_segmentation",
        lambda env: np.zeros((1, 6, 6), dtype=np.int64),
    )
    monkeypatch.setattr(
        oracle_logic,
        "_prepare_segmentation_visual",
        lambda seg, color_map, hw: (seg_vis, np.zeros((6, 6), dtype=np.int64)),
    )
    monkeypatch.setattr(
        oracle_logic,
        "_build_solve_options",
        lambda env, planner, selected_target, env_id: [],
    )

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    session.env = type(
        "_NoFrameEnv",
        (),
        {"unwrapped": _FakeUnwrapped(), "frames": [], "wrist_frames": []},
    )()
    session.planner = object()
    session.env_id = "BinFill"
    session.color_map = {}

    _img, msg = session.update_observation(use_segmentation=False)

    assert msg == "Ready"
    assert len(session.base_frames) == 0

    pil_img = session.get_pil_image(use_segmented=False)
    assert pil_img.size == (255, 255)


def test_update_observation_uses_only_front_rgb_list(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    monkeypatch.setattr(
        oracle_logic,
        "_fetch_segmentation",
        lambda env: np.zeros((1, 8, 8), dtype=np.int64),
    )
    monkeypatch.setattr(
        oracle_logic,
        "_build_solve_options",
        lambda env, planner, selected_target, env_id: [],
    )

    f1 = np.full((8, 8, 3), 11, dtype=np.uint8)
    f2 = np.full((8, 8, 3), 22, dtype=np.uint8)

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    session.env = _FakeObsWrapperEnv(front_rgb_list=[f1, f2], wrist_rgb_list=[])
    session.planner = object()
    session.env_id = "BinFill"
    session.color_map = {}

    _img, msg = session.update_observation(use_segmentation=False)

    assert msg == "Ready"
    assert len(session.base_frames) == 2
    assert len(session.wrist_frames) == 0
    assert session.base_frames[-1][0, 0, 0] == 22


def test_update_observation_does_not_duplicate_same_last_obs(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    monkeypatch.setattr(
        oracle_logic,
        "_fetch_segmentation",
        lambda env: np.zeros((1, 8, 8), dtype=np.int64),
    )
    monkeypatch.setattr(
        oracle_logic,
        "_build_solve_options",
        lambda env, planner, selected_target, env_id: [],
    )

    f1 = np.full((8, 8, 3), 10, dtype=np.uint8)
    f2 = np.full((8, 8, 3), 20, dtype=np.uint8)
    env = _FakeObsWrapperEnv(front_rgb_list=[f1, f2], wrist_rgb_list=[])

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    session.env = env
    session.planner = object()
    session.env_id = "BinFill"
    session.color_map = {}

    session.update_observation(use_segmentation=False)
    session.update_observation(use_segmentation=False)
    assert len(session.base_frames) == 2

    f3 = np.full((8, 8, 3), 30, dtype=np.uint8)
    env._last_obs = {"front_rgb_list": [f3], "wrist_rgb_list": []}
    session.update_observation(use_segmentation=False)
    assert len(session.base_frames) == 3
    assert session.base_frames[-1][0, 0, 0] == 30


def test_update_observation_does_not_fallback_to_env_frames(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    monkeypatch.setattr(
        oracle_logic,
        "_fetch_segmentation",
        lambda env: np.zeros((1, 8, 8), dtype=np.int64),
    )
    monkeypatch.setattr(
        oracle_logic,
        "_build_solve_options",
        lambda env, planner, selected_target, env_id: [],
    )

    env = _FakeEnv()
    env.frames = [np.full((8, 8, 3), 99, dtype=np.uint8)]

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    session.env = env
    session.planner = object()
    session.env_id = "BinFill"
    session.color_map = {}

    _img, msg = session.update_observation(use_segmentation=False)

    assert msg == "Ready"
    assert session.base_frames == []
