from __future__ import annotations

def test_load_next_task_wrapper_treats_episode98_as_normal(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    expected = ("SENTINEL",)

    monkeypatch.setattr(callbacks, "get_session", lambda uid: object())
    monkeypatch.setattr(
        callbacks.user_manager,
        "next_episode_same_env",
        lambda uid: {"is_done_all": False, "current_task": {"env_id": "BinFill", "episode_idx": 98}},
    )
    monkeypatch.setattr(callbacks, "_load_status_task", lambda uid, status: expected)

    result = callbacks.load_next_task_wrapper("uid1")

    assert result == expected


def test_restart_episode_wrapper_reloads_same_episode(monkeypatch, reload_module):
    reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    load_calls = []
    expected = ("RESTARTED",)

    monkeypatch.setattr(callbacks, "get_session", lambda uid: object())
    monkeypatch.setattr(
        callbacks.user_manager,
        "get_session_status",
        lambda uid: {"is_done_all": False, "current_task": {"env_id": "BinFill", "episode_idx": 98}},
    )

    def _fake_load_status_task(uid, status):
        load_calls.append((uid, status))
        return expected

    monkeypatch.setattr(callbacks, "_load_status_task", _fake_load_status_task)

    result = callbacks.restart_episode_wrapper("uid1")

    assert len(load_calls) == 1
    assert load_calls[0][1]["current_task"] == {"env_id": "BinFill", "episode_idx": 98}
    assert result == expected


def test_restart_episode_wrapper_missing_status_returns_login_failed(monkeypatch, reload_module):
    config = reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    monkeypatch.setattr(callbacks, "get_session", lambda uid: object())
    monkeypatch.setattr(callbacks.user_manager, "get_session_status", lambda uid: None)

    result = callbacks.restart_episode_wrapper("uid1")

    assert config.UI_TEXT["errors"]["restart_missing_task"] in result[3]


def test_execute_step_failed_episode98_still_advances(monkeypatch, reload_module):
    config = reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    class _FakeSession:
        def __init__(self):
            self.env_id = "BinFill"
            self.episode_idx = 98
            self.base_frames = []
            self.raw_solve_options = [{"available": False}]
            self.available_options = [("run", 0)]
            self.difficulty = "hard"
            self.language_goal = "goal"
            self.seed = 123
            self.non_demonstration_task_length = None

        def update_observation(self, use_segmentation=False):
            return None

        def get_pil_image(self, use_segmented=False):
            return "IMG"

        def execute_action(self, option_idx, click_coords):
            return "IMG", "FAILED", True

    fake_session = _FakeSession()
    complete_calls = []

    monkeypatch.setattr(callbacks, "get_session", lambda uid: fake_session)
    monkeypatch.setattr(callbacks, "increment_execute_count", lambda uid, env_id, episode_idx: 1)

    def _fake_complete_current_task(*args, **kwargs):
        payload = dict(kwargs)
        if args:
            payload["uid"] = args[0]
        complete_calls.append(payload)
        return {"is_done_all": False, "current_task": {"env_id": "MoveCube", "episode_idx": 7}}

    monkeypatch.setattr(callbacks.user_manager, "complete_current_task", _fake_complete_current_task)

    result = callbacks.execute_step("uid1", 0, config.UI_TEXT["coords"]["not_needed"])

    assert len(complete_calls) == 1
    assert complete_calls[0]["episode_idx"] == 98
    assert complete_calls[0]["status"] == "failed"
    assert result[2] == "BinFill (Episode 98)"
