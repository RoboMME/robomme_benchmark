from __future__ import annotations

import pytest
from PIL import Image


class _FakeOptionSession:
    def __init__(self, env_id="BinFill", raw_solve_options=None):
        self.env_id = env_id
        self.raw_solve_options = raw_solve_options or [{"label": "a", "available": True}]

    def get_pil_image(self, use_segmented=False):
        _ = use_segmented
        return Image.new("RGB", (8, 8), color=(0, 0, 0))


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


def test_on_option_select_uses_configured_select_point_and_log_messages(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setitem(callbacks.UI_TEXT["coords"], "select_point", "pick a point from config")
    monkeypatch.setitem(
        callbacks.UI_TEXT["log"],
        "point_selection_prompt",
        "custom log prompt from config",
    )
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeOptionSession())

    coords_text, img_update, log_text, suppress_flag, log_state = callbacks.on_option_select("uid-1", 0, None, False)

    assert coords_text == "pick a point from config"
    assert img_update.get("interactive") is True
    assert callbacks.get_live_obs_elem_classes(waiting_for_point=True) == img_update.get("elem_classes")
    assert log_text == "custom log prompt from config"
    assert suppress_flag is False
    assert log_state == callbacks._default_post_execute_log_state()


def test_on_map_click_uses_configured_selected_point_log(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setitem(
        callbacks.UI_TEXT["log"],
        "point_selected_message",
        "picked {label} @ <{x}, {y}>",
    )
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeOptionSession())

    event = type("Evt", (), {"index": (7, 9)})()
    _img, coords_text, log_text = callbacks.on_map_click("uid-1", 0, event)

    assert coords_text == "7, 9"
    assert log_text == "picked A @ <7, 9>"


def test_precheck_execute_inputs_uses_configured_before_execute_message(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setitem(callbacks.UI_TEXT["coords"], "select_point", "pick a point from config")
    monkeypatch.setitem(
        callbacks.UI_TEXT["coords"],
        "select_point_before_execute",
        "pick a point before execute from config",
    )
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeOptionSession())

    with pytest.raises(Exception) as excinfo:
        callbacks.precheck_execute_inputs("uid-1", 0, "pick a point from config")

    assert "pick a point before execute from config" in str(excinfo.value)


def test_on_video_end_transition_uses_configured_action_prompt(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setitem(callbacks.UI_TEXT["log"], "action_selection_prompt", "choose an action from config")

    result = callbacks.on_video_end_transition("uid-1", "demo_video")

    assert result[3] == "choose an action from config"
    assert result[4]["visible"] is False
    assert result[4]["interactive"] is False
    assert result[5] == "action_point"


def test_on_execute_video_end_transition_restores_controls_for_non_terminal_state(reload_module):
    callbacks = reload_module("gradio_callbacks")

    result = callbacks.on_execute_video_end_transition(
        "uid-1",
        {
            "exec_btn_interactive": True,
            "reference_action_interactive": True,
        },
        callbacks._default_post_execute_log_state(),
    )

    assert result[0]["visible"] is False
    assert result[1]["visible"] is True
    assert result[2]["visible"] is True
    assert result[3]["interactive"] is True
    assert result[4]["interactive"] is True
    assert result[5]["interactive"] is True
    assert result[6]["interactive"] is True
    assert result[8]["value"] == callbacks.UI_TEXT["log"]["action_selection_prompt"]
    assert result[9]["interactive"] is True
    assert result[10]["interactive"] is True
    assert result[11] == callbacks._default_post_execute_log_state()
    assert result[12] == "action_point"


def test_on_execute_video_end_transition_clears_execution_video_log_state(reload_module):
    callbacks = reload_module("gradio_callbacks")

    result = callbacks.on_execute_video_end_transition(
        "uid-1",
        {
            "exec_btn_interactive": True,
            "reference_action_interactive": True,
        },
        {
            "preserve_terminal_log": False,
            "terminal_log_value": None,
            "preserve_execute_video_log": True,
            "execute_video_log_value": "Executing: B",
        },
    )

    assert result[8]["value"] == callbacks.UI_TEXT["log"]["action_selection_prompt"]
    assert result[11] == callbacks._default_post_execute_log_state()
    assert result[12] == "action_point"


def test_on_execute_video_end_transition_keeps_terminal_buttons_disabled(reload_module):
    callbacks = reload_module("gradio_callbacks")

    result = callbacks.on_execute_video_end_transition(
        "uid-1",
        {
            "exec_btn_interactive": False,
            "reference_action_interactive": False,
        },
        {
            "preserve_terminal_log": True,
            "terminal_log_value": "terminal banner",
        },
    )

    assert result[0]["visible"] is False
    assert result[1]["visible"] is True
    assert result[2]["visible"] is True
    assert result[3]["interactive"] is True
    assert result[4]["interactive"] is False
    assert result[5]["interactive"] is True
    assert result[6]["interactive"] is True
    assert result[8]["value"] == "terminal banner"
    assert result[9]["interactive"] is False
    assert result[10]["interactive"] is True
    assert result[11] == callbacks._normalize_post_execute_log_state(
        {
            "preserve_terminal_log": True,
            "terminal_log_value": "terminal banner",
        }
    )
    assert result[12] == "action_point"


def test_on_option_select_preserves_terminal_log_state(reload_module):
    callbacks = reload_module("gradio_callbacks")

    coords_update, img_update, log_update, suppress_flag, log_state = callbacks.on_option_select(
        "uid-1",
        None,
        None,
        False,
        {
            "preserve_terminal_log": True,
            "terminal_log_value": "episode success banner",
        },
    )

    assert coords_update.get("__type__") == "update"
    assert img_update.get("__type__") == "update"
    assert log_update["value"] == "episode success banner"
    assert suppress_flag is False
    assert log_state == {
        "preserve_terminal_log": True,
        "terminal_log_value": "episode success banner",
        "preserve_execute_video_log": False,
        "execute_video_log_value": None,
    }


def test_on_option_select_preserves_execution_video_log_state(reload_module):
    callbacks = reload_module("gradio_callbacks")

    coords_update, img_update, log_update, suppress_flag, log_state = callbacks.on_option_select(
        "uid-1",
        1,
        callbacks.UI_TEXT["coords"]["select_point"],
        False,
        {
            "preserve_terminal_log": False,
            "terminal_log_value": None,
            "preserve_execute_video_log": True,
            "execute_video_log_value": "Executing: B",
        },
    )

    assert coords_update.get("__type__") == "update"
    assert img_update.get("__type__") == "update"
    assert log_update["value"] == "Executing: B"
    assert suppress_flag is False
    assert log_state == {
        "preserve_terminal_log": False,
        "terminal_log_value": None,
        "preserve_execute_video_log": True,
        "execute_video_log_value": "Executing: B",
    }


def test_on_demo_video_play_disables_button_and_sets_single_use_state(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")
    recorded = {"clicked": []}
    monkeypatch.setattr(callbacks, "get_session", lambda uid: object())
    monkeypatch.setattr(callbacks, "get_play_button_clicked", lambda uid: False)
    monkeypatch.setattr(
        callbacks,
        "set_play_button_clicked",
        lambda uid, clicked=True: recorded["clicked"].append((uid, clicked)),
    )

    result = callbacks.on_demo_video_play("uid-play")

    assert recorded["clicked"] == [("uid-play", True)]
    assert result["visible"] is True
    assert result["interactive"] is False


def test_missing_session_paths_use_configured_session_error(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setitem(callbacks.UI_TEXT["log"], "session_error", "Session Error From Config")
    monkeypatch.setattr(callbacks, "get_session", lambda uid: None)

    _img, _option_update, coords_text, log_text, suppress_flag = callbacks.on_reference_action("uid-missing", None)
    map_img, map_coords, map_log = callbacks.on_map_click("uid-missing", None, None)

    assert coords_text == callbacks.UI_TEXT["coords"]["not_needed"]
    assert log_text == "Session Error From Config"
    assert suppress_flag is False
    assert map_img.get("__type__") == "update"
    assert map_img.get("value") is None
    assert map_coords == callbacks.UI_TEXT["coords"]["not_needed"]
    assert map_log == "Session Error From Config"


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

    for raw_action, expected in patternlock_expected.items():
        assert config.get_ui_action_text("PatternLock", raw_action) == expected
    assert config.get_ui_action_text("BinFill", "pick up the cube") == "pick up the cube"


def test_ui_option_label_uses_patternlock_configured_action_text(reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")
    session = _FakeOptionSession(
        env_id="PatternLock",
        raw_solve_options=[{"label": "a", "action": "move forward", "available": False}],
    )

    assert callbacks._ui_option_label(session, "fallback", 0) == "A. move forward↓"


def test_load_status_task_appends_configured_point_suffix_after_mapped_label(monkeypatch, reload_module):
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
            f"A. move forward↓{config.UI_TEXT['actions']['point_required_suffix']}",
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
    assert result[8]["visible"] is False
    assert result[9]["visible"] is True
    assert result[9]["interactive"] is True
    assert result[15]["visible"] is True
    assert result[16]["visible"] is False
    assert result[17]["visible"] is False
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
    assert result[9]["visible"] is False
    assert result[9]["interactive"] is False
    assert result[15]["visible"] is False
    assert result[16]["visible"] is False
    assert result[17]["visible"] is True
    assert result[18]["visible"] is True
    assert callbacks.UI_TEXT["log"]["action_selection_prompt"] in result[3]


def test_load_status_task_uses_default_goal_format_for_videoplacebutton(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")
    session = _FakeLoadSession(
        env_id="VideoPlaceButton",
        available_options=[("pick", 0)],
        raw_solve_options=[{"label": "a", "action": "pick", "available": False}],
        language_goal="watch the video carefully, then place the red cube on the target right after the button was pressed",
    )

    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)
    monkeypatch.setattr(callbacks, "reset_play_button_clicked", lambda uid: None)
    monkeypatch.setattr(callbacks, "reset_execute_count", lambda uid, env_id, episode_idx: None)
    monkeypatch.setattr(callbacks, "set_task_start_time", lambda uid, env_id, episode_idx, start_time: None)
    monkeypatch.setattr(callbacks, "set_ui_phase", lambda uid, phase: None)
    monkeypatch.setattr(callbacks, "get_task_hint", lambda env_id: "")
    monkeypatch.setattr(callbacks, "should_show_demo_video", lambda env_id: False)

    result = callbacks._load_status_task(
        "uid-vpb",
        {"current_task": {"env_id": "VideoPlaceButton", "episode_idx": 1}, "completed_count": 0},
    )

    assert result[5] == (
        "Watch the video carefully, then place the red cube on the target right after the button was pressed"
    )


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
