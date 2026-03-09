from __future__ import annotations

import numpy as np


class _FakeSession:
    def __init__(self):
        self.env_id = "BinFill"
        self.episode_idx = 1
        self.raw_solve_options = [{"label": "a", "available": False}]
        self.available_options = [("pick", 0)]
        self.base_frames = []
        self.last_execution_frames = []
        self.non_demonstration_task_length = None
        self.difficulty = "easy"
        self.language_goal = "goal"
        self.seed = 123

    def get_pil_image(self, use_segmented=False):
        _ = use_segmented
        return "IMG"

    def update_observation(self, use_segmentation=False):
        _ = use_segmentation
        return None


def test_execute_step_builds_video_from_last_execution_frames(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")

    frame1 = np.full((8, 8, 3), 11, dtype=np.uint8)
    frame2 = np.full((8, 8, 3), 22, dtype=np.uint8)
    session = _FakeSession()
    session.base_frames = [frame2]

    def _execute_action(_option_idx, _coords):
        session.last_execution_frames = [frame1, frame2]
        return "IMG", "Executing: pick", False

    session.execute_action = _execute_action

    captured = {}
    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)
    monkeypatch.setattr(callbacks, "increment_execute_count", lambda uid, env_id, episode_idx: 1)
    monkeypatch.setattr(callbacks, "concatenate_frames_horizontally", lambda frames, env_id=None: list(frames))
    def _save_video(frames, suffix=""):
        captured["payload"] = (list(frames), suffix)
        return "/tmp/exec.mp4"

    monkeypatch.setattr(callbacks, "save_video", _save_video)
    monkeypatch.setattr(callbacks.os.path, "exists", lambda path: True)
    monkeypatch.setattr(callbacks.os.path, "getsize", lambda path: 10)

    result = callbacks.execute_step("uid-1", 0, callbacks.UI_TEXT["coords"]["not_needed"])

    saved_frames, suffix = captured["payload"]
    assert [int(frame[0, 0, 0]) for frame in saved_frames] == [11, 22]
    assert suffix.startswith("execute_")
    assert result[7]["visible"] is True
    assert result[8]["visible"] is False
    assert result[9]["visible"] is True
    assert result[10]["visible"] is True
    assert result[11]["value"] is None
    assert result[11]["interactive"] is False
    assert result[14]["interactive"] is False
    expected_log = callbacks.UI_TEXT["log"]["execute_action_prompt"].format(label="A")
    assert result[1] == expected_log
    assert result[15] == {
        "exec_btn_interactive": True,
        "reference_action_interactive": True,
    }
    assert result[16] == {
        "preserve_terminal_log": False,
        "terminal_log_value": None,
        "preserve_execute_video_log": True,
        "execute_video_log_value": expected_log,
    }
    assert result[17] == "execution_video"


def test_execute_step_execution_log_includes_point_when_coords_selected(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")

    frame = np.full((8, 8, 3), 44, dtype=np.uint8)
    session = _FakeSession()
    session.raw_solve_options = [{"label": "b", "available": [object()]}]
    session.base_frames = [frame]

    captured = {}

    def _execute_action(_option_idx, coords):
        captured["coords"] = coords
        session.last_execution_frames = [frame]
        return "IMG", "Executing: pick", False

    session.execute_action = _execute_action

    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)
    monkeypatch.setattr(callbacks, "increment_execute_count", lambda uid, env_id, episode_idx: 1)
    monkeypatch.setattr(callbacks, "concatenate_frames_horizontally", lambda frames, env_id=None: list(frames))
    monkeypatch.setattr(callbacks, "save_video", lambda frames, suffix="": "/tmp/exec-point.mp4")
    monkeypatch.setattr(callbacks.os.path, "exists", lambda path: True)
    monkeypatch.setattr(callbacks.os.path, "getsize", lambda path: 10)

    result = callbacks.execute_step("uid-1", 0, "12, 34")

    assert captured["coords"] == (12, 34)
    assert result[1] == "Executing: B | point <12, 34>"
    assert result[16]["execute_video_log_value"] == "Executing: B | point <12, 34>"


def test_execute_step_falls_back_to_single_frame_clip_when_no_new_frames(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")

    frame = np.full((8, 8, 3), 33, dtype=np.uint8)
    session = _FakeSession()
    session.base_frames = [frame]

    def _execute_action(_option_idx, _coords):
        session.last_execution_frames = []
        return "IMG", "Executing: pick", False

    session.execute_action = _execute_action

    captured = {}
    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)
    monkeypatch.setattr(callbacks, "increment_execute_count", lambda uid, env_id, episode_idx: 1)
    monkeypatch.setattr(callbacks, "concatenate_frames_horizontally", lambda frames, env_id=None: list(frames))
    def _save_video(frames, suffix=""):
        captured["frames"] = list(frames)
        return "/tmp/exec-single.mp4"

    monkeypatch.setattr(callbacks, "save_video", _save_video)
    monkeypatch.setattr(callbacks.os.path, "exists", lambda path: True)
    monkeypatch.setattr(callbacks.os.path, "getsize", lambda path: 10)

    result = callbacks.execute_step("uid-1", 0, callbacks.UI_TEXT["coords"]["not_needed"])

    assert len(captured["frames"]) == 1
    assert int(captured["frames"][0][0, 0, 0]) == 33
    assert result[7]["visible"] is True
    assert result[10]["visible"] is True
    assert result[17] == "execution_video"


def test_switch_phase_toggles_live_obs_interactive_without_refresh_queue(reload_module):
    config = reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    to_exec = callbacks.switch_to_execute_phase("uid-3")
    assert len(to_exec) == 7
    assert to_exec[0].get("interactive") is False
    assert to_exec[4].get("interactive") is False
    assert to_exec[4].get("elem_classes") == config.get_live_obs_elem_classes()
    assert to_exec[5].get("interactive") is False
    assert to_exec[6].get("interactive") is False

    to_action = callbacks.switch_to_action_phase()
    assert len(to_action) == 6
    assert to_action[0].get("interactive") is True
    assert to_action[4].get("interactive") is True
    assert to_action[4].get("elem_classes") == config.get_live_obs_elem_classes()
    assert to_action[5].get("interactive") is True
