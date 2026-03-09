from __future__ import annotations

from PIL import Image


class _FakeSession:
    def __init__(self, reference_payload, env_id="BinFill"):
        self._reference_payload = reference_payload
        self.env_id = env_id

    def get_reference_action(self):
        return self._reference_payload

    def get_pil_image(self, use_segmented=True):
        return Image.new("RGB", (24, 24), color=(0, 0, 0))


class _FakeOptionSession:
    def __init__(self):
        self.raw_solve_options = [{"available": [object()]}]
        self.available_options = [("pick", 0)]

    def get_pil_image(self, use_segmented=True):
        _ = use_segmented
        return Image.new("RGB", (24, 24), color=(0, 0, 0))


def _is_fluorescent_yellow(pixel):
    return pixel[0] > 180 and pixel[1] > 200 and pixel[2] < 80


def test_on_reference_action_success_updates_option_and_coords(monkeypatch, reload_module):
    config = reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    session = _FakeSession(
        {
            "ok": True,
            "option_idx": 2,
            "option_label": "c",
            "option_action": "press the button",
            "need_coords": True,
            "coords_xy": [5, 6],
            "message": "ok",
        }
    )

    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)

    img_update, option_update, coords_text, log_html, suppress_flag = callbacks.on_reference_action("uid-1", None)

    assert img_update.get("__type__") == "update"
    assert isinstance(img_update.get("value"), Image.Image)
    assert _is_fluorescent_yellow(img_update["value"].getpixel((5, 6)))
    assert img_update.get("elem_classes") == config.get_live_obs_elem_classes()
    assert option_update.get("value") == 2
    assert coords_text == "5, 6"
    assert suppress_flag is True
    expected_log = config.UI_TEXT["log"]["reference_action_message_with_coords"].format(
        option_label="c",
        option_action="press the button",
        coords_text="5, 6",
    )
    assert log_html == expected_log


def test_on_reference_action_session_missing(monkeypatch, reload_module):
    config = reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setattr(callbacks, "get_session", lambda uid: None)

    img_update, option_update, coords_text, log_html, suppress_flag = callbacks.on_reference_action("uid-missing", None)

    assert img_update.get("__type__") == "update"
    assert img_update.get("value") is None
    assert option_update.get("__type__") == "update"
    assert coords_text == config.UI_TEXT["coords"]["not_needed"]
    assert log_html == config.UI_TEXT["log"]["session_error"]
    assert suppress_flag is False


def test_on_reference_action_error_message_from_reference(monkeypatch, reload_module):
    config = reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    session = _FakeSession({"ok": False, "message": "bad ref"})
    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)

    _img, _opt, _coords, log_html, suppress_flag = callbacks.on_reference_action("uid-1", None)
    assert log_html == config.UI_TEXT["log"]["reference_action_status"].format(message="bad ref")
    assert suppress_flag is False


def test_on_reference_action_same_selected_option_does_not_set_suppression(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")

    session = _FakeSession(
        {
            "ok": True,
            "option_idx": 0,
            "option_label": "a",
            "option_action": "pick up the cube",
            "need_coords": True,
            "coords_xy": [3, 4],
            "message": "ok",
        }
    )

    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)

    _img, _option_update, coords_text, _log_html, suppress_flag = callbacks.on_reference_action("uid-1", 0)

    assert coords_text == "3, 4"
    assert suppress_flag is False


def test_on_option_select_resets_to_point_wait_state_for_point_action(monkeypatch, reload_module):
    config = reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    session = _FakeOptionSession()
    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)

    coords_text, img_update, log_text, suppress_flag, log_state = callbacks.on_option_select("uid-1", 0, "12, 34", False)

    assert coords_text == config.UI_TEXT["coords"]["select_point"]
    assert img_update.get("interactive") is True
    assert img_update.get("elem_classes") == config.get_live_obs_elem_classes(waiting_for_point=True)
    assert log_text == config.UI_TEXT["log"]["point_selection_prompt"]
    assert suppress_flag is False
    assert log_state == callbacks._default_post_execute_log_state()


def test_on_option_select_suppresses_programmatic_reference_change(reload_module):
    callbacks = reload_module("gradio_callbacks")

    coords_update, img_update, log_update, suppress_flag, log_state = callbacks.on_option_select("uid-1", 0, "12, 34", True)

    assert coords_update.get("__type__") == "update"
    assert img_update.get("__type__") == "update"
    assert log_update.get("__type__") == "update"
    assert suppress_flag is False
    assert log_state == callbacks._default_post_execute_log_state()


def test_on_map_click_clears_wait_state_and_restores_action_prompt(monkeypatch, reload_module):
    config = reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    session = _FakeOptionSession()
    event = type("Evt", (), {"index": (5, 6)})()
    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)

    img_update, coords_text, log_text = callbacks.on_map_click("uid-1", 0, event)

    assert img_update.get("__type__") == "update"
    assert isinstance(img_update.get("value"), Image.Image)
    assert _is_fluorescent_yellow(img_update["value"].getpixel((5, 6)))
    assert img_update.get("elem_classes") == config.get_live_obs_elem_classes()
    assert coords_text == "5, 6"
    assert log_text == config.UI_TEXT["log"]["action_selection_prompt"]


def test_on_reference_action_uses_configured_action_text_override(monkeypatch, reload_module):
    config = reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    session = _FakeSession(
        {
            "ok": True,
            "option_idx": 0,
            "option_label": "a",
            "option_action": "move forward",
            "need_coords": False,
            "coords_xy": None,
            "message": "ok",
        },
        env_id="PatternLock",
    )

    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)

    _img, _option_update, coords_text, log_html, suppress_flag = callbacks.on_reference_action("uid-1", None)

    assert coords_text == config.UI_TEXT["coords"]["not_needed"]
    assert log_html == config.UI_TEXT["log"]["reference_action_message"].format(
        option_label="a",
        option_action="move forward↓",
    )
    assert suppress_flag is True
