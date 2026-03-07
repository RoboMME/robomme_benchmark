from __future__ import annotations

import pytest


class _FakeOptionSession:
    def __init__(self, env_id="BinFill", raw_solve_options=None):
        self.env_id = env_id
        self.raw_solve_options = raw_solve_options or [{"available": True}]


class _FakeLoadSession:
    def __init__(self, env_id, available_options, raw_solve_options, demonstration_frames=None, language_goal=""):
        self.env_id = env_id
        self.available_options = available_options
        self.raw_solve_options = raw_solve_options
        self.language_goal = language_goal
        self.demonstration_frames = demonstration_frames or []

    def load_episode(self, env_id, episode_idx):
        self.env_id = env_id
        return "IMG", f"loaded {env_id} {episode_idx}"

    def get_pil_image(self, use_segmented=False):
        return "IMG"


def test_on_option_select_uses_configured_select_keypoint_message(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setitem(callbacks.UI_TEXT["coords"], "select_keypoint", "pick a point from config")
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeOptionSession())

    coords_text, img_update = callbacks.on_option_select("uid-1", 0, None)

    assert coords_text == "pick a point from config"
    assert img_update.get("interactive") is True


def test_precheck_execute_inputs_uses_configured_before_execute_message(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setitem(callbacks.UI_TEXT["coords"], "select_keypoint", "pick a point from config")
    monkeypatch.setitem(
        callbacks.UI_TEXT["coords"],
        "select_keypoint_before_execute",
        "pick a point before execute from config",
    )
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeOptionSession())

    with pytest.raises(Exception) as excinfo:
        callbacks.precheck_execute_inputs("uid-1", 0, "pick a point from config")

    assert "pick a point before execute from config" in str(excinfo.value)


def test_on_video_end_transition_uses_configured_action_prompt(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setitem(callbacks.UI_TEXT["log"], "action_selection_prompt", "choose an action from config")

    result = callbacks.on_video_end_transition("uid-1")

    assert result[3] == "choose an action from config"
    assert result[4]["visible"] is False
    assert result[4]["interactive"] is False


def test_on_demo_video_play_disables_button_and_sets_single_use_state(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")
    recorded = {"activity": [], "clicked": []}

    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: recorded["activity"].append(uid))
    monkeypatch.setattr(callbacks, "get_play_button_clicked", lambda uid: False)
    monkeypatch.setattr(
        callbacks,
        "set_play_button_clicked",
        lambda uid, clicked=True: recorded["clicked"].append((uid, clicked)),
    )

    result = callbacks.on_demo_video_play("uid-play")

    assert recorded["activity"] == ["uid-play"]
    assert recorded["clicked"] == [("uid-play", True)]
    assert result["visible"] is True
    assert result["interactive"] is False


def test_missing_session_paths_use_configured_session_error(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setitem(callbacks.UI_TEXT["log"], "session_error", "Session Error From Config")
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: None)

    _img, _option_update, coords_text, log_text = callbacks.on_reference_action("uid-missing")
    map_img, map_text = callbacks.on_map_click("uid-missing", None, None)

    assert coords_text == callbacks.UI_TEXT["coords"]["not_needed"]
    assert log_text == "Session Error From Config"
    assert map_img is None
    assert map_text == "Session Error From Config"


def test_get_ui_action_text_uses_configured_overrides_and_fallback(reload_module):
    config = reload_module("config")

    patternlock_expected = {
        "move forward": "move forward↓",
        "move backward": "move backward↑",
        "move left": "move left→",
        "move right": "move right←",
        "move forward-left": "move forward-left↘︎",
        "move forward-right": "move forward-right↙︎",
        "move backward-left": "move backward-left↗︎",
        "move backward-right": "move backward-right↖︎",
    }
    routestick_expected = {
        "move to the nearest left target by circling around the stick clockwise": "move left clockwise↘︎→↗︎ ◟→◞",
        "move to the nearest right target by circling around the stick clockwise": "move right clockwise↖︎←↙︎ ◟←◞",
        "move to the nearest left target by circling around the stick counterclockwise": "move left counterclockwise↗︎→↘︎ ◜→◝",
        "move to the nearest right target by circling around the stick counterclockwise": "move right counterclockwise↙︎←↖︎ ◜←◝",
    }

    for raw_action, expected in patternlock_expected.items():
        assert config.get_ui_action_text("PatternLock", raw_action) == expected
    for raw_action, expected in routestick_expected.items():
        assert config.get_ui_action_text("RouteStick", raw_action) == expected
    assert config.get_ui_action_text("BinFill", "pick up the cube") == "pick up the cube"


def test_ui_option_label_uses_patternlock_configured_action_text(reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")
    session = _FakeOptionSession(
        env_id="PatternLock",
        raw_solve_options=[{"label": "a", "action": "move forward", "available": False}],
    )

    assert callbacks._ui_option_label(session, "fallback", 0) == "a. move forward↓"


def test_ui_option_label_uses_routestick_configured_action_text(reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")
    session = _FakeOptionSession(
        env_id="RouteStick",
        raw_solve_options=[
            {
                "label": "d",
                "action": "move to the nearest right target by circling around the stick counterclockwise",
                "available": False,
            }
        ],
    )

    assert callbacks._ui_option_label(session, "fallback", 0) == "d. move right counterclockwise↙︎←↖︎ ◜←◝"


def test_load_status_task_appends_configured_keypoint_suffix_after_mapped_label(monkeypatch, reload_module):
    config = reload_module("config")
    callbacks = reload_module("gradio_callbacks")
    session = _FakeLoadSession(
        env_id="PatternLock",
        available_options=[("a. move forward", 0)],
        raw_solve_options=[{"label": "a", "action": "move forward", "available": [object()]}],
    )

    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)
    monkeypatch.setattr(callbacks, "reset_play_button_clicked", lambda uid: None)
    monkeypatch.setattr(callbacks, "reset_execute_count", lambda uid, env_id, episode_idx: None)
    monkeypatch.setattr(callbacks, "set_task_start_time", lambda uid, env_id, episode_idx, start_time: None)
    monkeypatch.setattr(callbacks, "set_ui_phase", lambda uid, phase: None)
    monkeypatch.setattr(callbacks, "get_task_hint", lambda env_id: "")
    monkeypatch.setattr(callbacks, "should_show_demo_video", lambda env_id: False)

    result = callbacks._load_status_task(
        "uid-1",
        {"current_task": {"env_id": "PatternLock", "episode_idx": 1}, "completed_count": 3},
    )

    assert result[4]["choices"] == [
        (
            f"a. move forward↓{config.UI_TEXT['actions']['keypoint_required_suffix']}",
            0,
        )
    ]


def test_load_status_task_shows_demo_video_button_for_valid_video(monkeypatch, reload_module, tmp_path):
    callbacks = reload_module("gradio_callbacks")
    session = _FakeLoadSession(
        env_id="VideoUnmask",
        available_options=[("pick", 0)],
        raw_solve_options=[{"label": "a", "action": "pick", "available": False}],
        demonstration_frames=["frame-1"],
        language_goal="remember the cube",
    )
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")

    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)
    monkeypatch.setattr(callbacks, "reset_play_button_clicked", lambda uid: None)
    monkeypatch.setattr(callbacks, "reset_execute_count", lambda uid, env_id, episode_idx: None)
    monkeypatch.setattr(callbacks, "set_task_start_time", lambda uid, env_id, episode_idx, start_time: None)
    monkeypatch.setattr(callbacks, "set_ui_phase", lambda uid, phase: None)
    monkeypatch.setattr(callbacks, "get_task_hint", lambda env_id: "")
    monkeypatch.setattr(callbacks, "should_show_demo_video", lambda env_id: True)
    monkeypatch.setattr(callbacks, "save_video", lambda frames, suffix="": str(video_path))

    result = callbacks._load_status_task(
        "uid-video",
        {"current_task": {"env_id": "VideoUnmask", "episode_idx": 1}, "completed_count": 0},
    )

    assert result[7]["visible"] is True
    assert result[7]["value"] == str(video_path)
    assert result[8]["visible"] is True
    assert result[8]["interactive"] is True
    assert result[14]["visible"] is True
    assert result[15]["visible"] is False
    assert result[16]["visible"] is False
    assert callbacks.UI_TEXT["log"]["demo_video_prompt"] in result[3]


def test_load_status_task_hides_demo_video_button_when_video_is_missing(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")
    session = _FakeLoadSession(
        env_id="VideoUnmask",
        available_options=[("pick", 0)],
        raw_solve_options=[{"label": "a", "action": "pick", "available": False}],
        demonstration_frames=["frame-1"],
        language_goal="remember the cube",
    )

    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)
    monkeypatch.setattr(callbacks, "reset_play_button_clicked", lambda uid: None)
    monkeypatch.setattr(callbacks, "reset_execute_count", lambda uid, env_id, episode_idx: None)
    monkeypatch.setattr(callbacks, "set_task_start_time", lambda uid, env_id, episode_idx, start_time: None)
    monkeypatch.setattr(callbacks, "set_ui_phase", lambda uid, phase: None)
    monkeypatch.setattr(callbacks, "get_task_hint", lambda env_id: "")
    monkeypatch.setattr(callbacks, "should_show_demo_video", lambda env_id: True)
    monkeypatch.setattr(callbacks, "save_video", lambda frames, suffix="": None)

    result = callbacks._load_status_task(
        "uid-no-video",
        {"current_task": {"env_id": "VideoUnmask", "episode_idx": 1}, "completed_count": 0},
    )

    assert result[7]["visible"] is False
    assert result[8]["visible"] is False
    assert result[8]["interactive"] is False
    assert result[14]["visible"] is False
    assert result[15]["visible"] is True
    assert result[16]["visible"] is True
    assert callbacks.UI_TEXT["log"]["action_selection_prompt"] in result[3]


def test_draw_coordinate_axes_uses_configured_routestick_overlay_labels(monkeypatch, reload_module):
    config = reload_module("config")
    image_utils = reload_module("image_utils")
    recorded_texts = []
    original_text = image_utils.ImageDraw.ImageDraw.text

    def _record_text(self, xy, text, *args, **kwargs):
        recorded_texts.append(text)
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(image_utils.ImageDraw.ImageDraw, "text", _record_text)

    img = image_utils.Image.new("RGB", (220, 260), color=(0, 0, 0))
    image_utils.draw_coordinate_axes(img, position="left", env_id="RouteStick")

    expected_labels = [
        config.get_ui_action_text("RouteStick", action_text)
        for action_text in config.ROUTESTICK_OVERLAY_ACTION_TEXTS
    ]
    for label in expected_labels:
        assert label in recorded_texts
