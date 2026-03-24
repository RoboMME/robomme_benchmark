from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._shared.repo_paths import ensure_src_on_path

ensure_src_on_path(__file__)

from robomme.env_record_wrapper.RecordWrapper import RobommeRecordWrapper
from robomme.env_record_wrapper.episode_object_logging import (
    EPISODE_OBJECT_LOG_FILENAME,
    init_episode_object_log_state,
    record_reset_objects,
)


pytestmark = [pytest.mark.lightweight]


class _DummyEnv(gym.Env):
    metadata = {}
    observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self):
        super().__init__()
        self.use_demonstrationwrapper = False
        self.spec = SimpleNamespace(id="DummyEnv")
        init_episode_object_log_state(self)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}


class _Pose:
    def __init__(self, position):
        self.p = position


class _Actor:
    def __init__(self, name: str, position):
        self.name = name
        self.pose = _Pose(position)


@pytest.mark.parametrize("populate_state", [True, False])
def test_record_wrapper_close_writes_minimal_episode_object_log_jsonl(
    tmp_path,
    monkeypatch,
    populate_state,
):
    record_wrapper_mod = importlib.import_module("robomme.env_record_wrapper.RecordWrapper")
    monkeypatch.setattr(record_wrapper_mod.task_goal, "get_language_goal", lambda *args, **kwargs: [])
    monkeypatch.setattr(RobommeRecordWrapper, "_video_flush_episode_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(RobommeRecordWrapper, "_contact_check_flush", lambda *args, **kwargs: None)

    env = _DummyEnv()
    if populate_state:
        record_reset_objects(
            env,
            bin_list=[{"actor": _Actor("bin_0", [0.1, 0.0, 0.04]), "color": "red"}],
            cube_list=[{"actor": _Actor("cube_red", [0.1, 0.0, 0.02]), "color": "red"}],
            target_cube_list=[{"actor": _Actor("cube_green", [0.0, 0.1, 0.02]), "color": "green"}],
        )

    wrapper = RobommeRecordWrapper(
        env,
        dataset=str(tmp_path),
        env_id="DummyEnv",
        episode=7,
        seed=11,
        save_video=False,
        record_hdf5=False,
    )
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
    assert record == {
        "env": "DummyEnv",
        "episode": 7,
        "seed": 11,
        "bin_list": (
            [{"name": "bin_0", "position": [0.1, 0.0, 0.04], "color": "red"}]
            if populate_state
            else []
        ),
        "cube_list": (
            [{"name": "cube_red", "position": [0.1, 0.0, 0.02], "color": "red"}]
            if populate_state
            else []
        ),
        "target_cube_list": (
            [{"name": "cube_green", "position": [0.0, 0.1, 0.02], "color": "green"}]
            if populate_state
            else []
        ),
        "swap_events": [],
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
