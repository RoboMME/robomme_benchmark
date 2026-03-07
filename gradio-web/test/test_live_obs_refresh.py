from __future__ import annotations

import numpy as np
from PIL import Image


class _FakeSession:
    def __init__(self, frames, env_id="BinFill"):
        self.base_frames = frames
        self.env_id = env_id


def test_refresh_live_obs_skips_when_not_execution_phase(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")
    monkeypatch.setattr(callbacks, "get_session", lambda uid: _FakeSession([]))

    update = callbacks.refresh_live_obs("uid-1", "action_keypoint")

    assert update.get("__type__") == "update"
    assert "value" not in update


def test_refresh_live_obs_updates_image_from_latest_frame(monkeypatch, reload_module):
    config = reload_module("config")
    callbacks = reload_module("gradio_callbacks")
    frame0 = np.zeros((8, 8, 3), dtype=np.uint8)
    frame1 = np.full((8, 8, 3), 11, dtype=np.uint8)
    frame2 = np.full((8, 8, 3), 22, dtype=np.uint8)
    frame3 = np.full((8, 8, 3), 33, dtype=np.uint8)
    frame4 = np.full((8, 8, 3), 44, dtype=np.uint8)
    session = _FakeSession([frame0])
    monkeypatch.setattr(callbacks, "get_session", lambda uid: session)
    monkeypatch.setattr(callbacks, "KEYFRAME_DOWNSAMPLE_FACTOR", 2)

    # Reset queue state at execute start (cursor anchored at current base_frames length).
    callbacks.switch_to_execute_phase("uid-2")
    session.base_frames.extend([frame1, frame2, frame3, frame4])

    # Downsample x2 + FIFO => first frame1, then frame3.
    update1 = callbacks.refresh_live_obs("uid-2", "execution_playback")
    update2 = callbacks.refresh_live_obs("uid-2", "execution_playback")
    update3 = callbacks.refresh_live_obs("uid-2", "execution_playback")

    assert update1.get("__type__") == "update"
    assert update1.get("interactive") is False
    assert update1.get("elem_classes") == config.get_live_obs_elem_classes()
    assert isinstance(update1.get("value"), Image.Image)
    assert update1["value"].getpixel((0, 0)) == (11, 11, 11)

    assert update2.get("__type__") == "update"
    assert update2.get("interactive") is False
    assert update2.get("elem_classes") == config.get_live_obs_elem_classes()
    assert isinstance(update2.get("value"), Image.Image)
    assert update2["value"].getpixel((0, 0)) == (33, 33, 33)

    # Queue drained, so no further value update.
    assert update3.get("__type__") == "update"
    assert "value" not in update3


def test_switch_phase_keeps_live_obs_visible_and_toggles_interactive(reload_module):
    config = reload_module("config")
    callbacks = reload_module("gradio_callbacks")

    to_exec = callbacks.switch_to_execute_phase("uid-3")
    assert len(to_exec) == 6
    assert to_exec[0].get("interactive") is False
    assert to_exec[4].get("interactive") is False
    assert to_exec[4].get("elem_classes") == config.get_live_obs_elem_classes()
    assert to_exec[5].get("interactive") is False

    to_action = callbacks.switch_to_action_phase()
    assert len(to_action) == 6
    assert to_action[0].get("interactive") is True
    assert to_action[4].get("interactive") is True
    assert to_action[4].get("elem_classes") == config.get_live_obs_elem_classes()
    assert to_action[5].get("interactive") is True
