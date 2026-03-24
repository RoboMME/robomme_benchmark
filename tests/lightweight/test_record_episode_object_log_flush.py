from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._shared.repo_paths import ensure_src_on_path
ensure_src_on_path(__file__)

from robomme.env_record_wrapper.RecordWrapper import RobommeRecordWrapper
from robomme.env_record_wrapper.episode_object_logging import EPISODE_OBJECT_LOG_FILENAME


pytestmark = [pytest.mark.lightweight]


class _DummyEnv(gym.Env):
    metadata = {}
    observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self, *, object_log=None, expose_method: bool = True):
        super().__init__()
        self.use_demonstrationwrapper = False
        self.difficulty = "easy"
        self.spec = SimpleNamespace(id="DummyEnv")
        self._object_log = object_log
        if expose_method:
            self.get_episode_object_log = lambda: self._object_log

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}


@pytest.mark.parametrize("expose_method", [True, False])
def test_record_wrapper_close_writes_episode_object_log_jsonl(tmp_path, monkeypatch, expose_method):
    record_wrapper_mod = importlib.import_module("robomme.env_record_wrapper.RecordWrapper")
    monkeypatch.setattr(record_wrapper_mod.task_goal, "get_language_goal", lambda *args, **kwargs: [])
    monkeypatch.setattr(RobommeRecordWrapper, "_video_flush_episode_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(RobommeRecordWrapper, "_contact_check_flush", lambda *args, **kwargs: None)

    object_log = {
        "cube_bins": [
            {
                "bin_index": 0,
                "color": "red",
                "bin_world_position": [0.1, 0.0, 0.04],
                "cube_world_position": [0.1, 0.0, 0.02],
            }
        ],
        "target_cube": {
            "type": "cube",
            "name": "target_cube_red",
            "actor_name": "target_cube_red",
            "color": "red",
            "bin_index": 0,
            "world_position": [0.1, 0.0, 0.02],
        },
        "swap_events": [],
    }
    env = _DummyEnv(object_log=object_log, expose_method=expose_method)
    wrapper = RobommeRecordWrapper(
        env,
        dataset=str(tmp_path),
        env_id="DummyEnv",
        episode=7,
        seed=11,
        save_video=False,
        record_hdf5=False,
    )
    wrapper.episode_success = True
    wrapper.close()

    jsonl_path = tmp_path / EPISODE_OBJECT_LOG_FILENAME
    assert jsonl_path.exists()
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == 1
    assert record["env"] == "DummyEnv"
    assert record["episode"] == 7
    assert record["seed"] == 11
    assert record["difficulty"] == "easy"
    assert record["episode_success"] is True
    if expose_method:
        assert record["object_log"]["target_cube"]["name"] == "target_cube_red"
        assert record["object_log"]["cube_bins"][0]["bin_world_position"] == [0.1, 0.0, 0.04]
    else:
        assert record["object_log"] == {
            "cube_bins": [],
            "target_cube": None,
            "swap_events": [],
        }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
