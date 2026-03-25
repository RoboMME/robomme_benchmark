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
from robomme.env_record_wrapper import object_log as objectlog
from robomme.robomme_env.utils import swap_contact_monitoring as swapContact


pytestmark = [pytest.mark.lightweight]


class _DummyEnv(gym.Env):
    metadata = {}
    observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self):
        super().__init__()
        self.use_demonstrationwrapper = False
        self.spec = SimpleNamespace(id="DummyEnv")
        objectlog.init_episode_log(self)

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

    env = _DummyEnv()
    if populate_state:
        objectlog.record_object(
            env,
            event="reset",
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

    jsonl_path = tmp_path / objectlog.EPISODE_OBJECT_LOG_FILENAME
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
        "collision_events": [],
    }


def test_record_wrapper_close_writes_collision_summary_into_episode_object_log(
    tmp_path,
    monkeypatch,
):
    record_wrapper_mod = importlib.import_module("robomme.env_record_wrapper.RecordWrapper")
    monkeypatch.setattr(record_wrapper_mod.task_goal, "get_language_goal", lambda *args, **kwargs: [])
    monkeypatch.setattr(RobommeRecordWrapper, "_video_flush_episode_files", lambda *args, **kwargs: None)

    env = _DummyEnv()
    env.swap_contact_state = swapContact.new_swap_contact_state()
    env.swap_contact_state.swap_contact_detected = True
    env.swap_contact_state.first_contact_step = 224
    env.swap_contact_state.contact_pairs.append("bin_1<->bin_2")
    env.swap_contact_state.max_force_norm = 0.010015421144429798
    env.swap_contact_state.max_force_pair = "bin_1<->bin_2"
    env.swap_contact_state.max_force_step = 224
    env.swap_contact_state.pair_max_force["bin_1<->bin_2"] = 0.010015421144429798

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

    jsonl_path = tmp_path / objectlog.EPISODE_OBJECT_LOG_FILENAME
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["collision_events"] == [
        {
            "swap_contact_detected": True,
            "first_contact_step": 224,
            "contact_pairs": ["bin_1<->bin_2"],
            "max_force_norm": 0.010015421144429798,
            "max_force_pair": "bin_1<->bin_2",
            "max_force_step": 224,
            "pair_max_force": {"bin_1<->bin_2": 0.010015421144429798},
        }
    ]
    built_record = objectlog.build_episode_object_log_record(
        env,
        env_id="DummyEnv",
        episode=7,
        seed=11,
    )
    assert built_record["collision_events"] == []
    assert not (tmp_path / "contact_check_results.jsonl").exists()


def test_flush_episode_log_appends_collision_summary_without_mutating_env_state(tmp_path):
    env = _DummyEnv()
    objectlog.record_object(
        env,
        event="reset",
        bin_list=[{"actor": _Actor("bin_0", [0.1, 0.0, 0.04]), "color": "red"}],
        cube_list=[],
        target_cube_list=[],
    )

    jsonl_path = objectlog.flush_episode_log(
        env,
        output_root=tmp_path,
        env_id="DummyEnv",
        episode=7,
        seed=11,
        contact_summary={
            "swap_contact_detected": True,
            "first_contact_step": 224,
            "contact_pairs": ["bin_1<->bin_2"],
            "max_force_norm": 0.010015421144429798,
            "max_force_pair": "bin_1<->bin_2",
            "max_force_step": 224,
            "pair_max_force": {"bin_1<->bin_2": 0.010015421144429798},
        },
    )

    assert jsonl_path == tmp_path / objectlog.EPISODE_OBJECT_LOG_FILENAME
    record = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["collision_events"] == [
        {
            "swap_contact_detected": True,
            "first_contact_step": 224,
            "contact_pairs": ["bin_1<->bin_2"],
            "max_force_norm": 0.010015421144429798,
            "max_force_pair": "bin_1<->bin_2",
            "max_force_step": 224,
            "pair_max_force": {"bin_1<->bin_2": 0.010015421144429798},
        }
    ]
    built_record = objectlog.build_episode_object_log_record(
        env,
        env_id="DummyEnv",
        episode=7,
        seed=11,
    )
    assert built_record["collision_events"] == []


def test_record_wrapper_does_not_prefix_video_filename_when_collision_detected(
    tmp_path,
    monkeypatch,
):
    record_wrapper_mod = importlib.import_module("robomme.env_record_wrapper.RecordWrapper")
    monkeypatch.setattr(record_wrapper_mod.task_goal, "get_language_goal", lambda *args, **kwargs: [])

    def _fake_video_write(self, frames, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"video")

    monkeypatch.setattr(RobommeRecordWrapper, "_video_write_mp4", _fake_video_write)

    env = _DummyEnv()
    env.swap_contact_state = swapContact.new_swap_contact_state()
    env.swap_contact_state.swap_contact_detected = True
    env.swap_contact_state.first_contact_step = 12
    env.swap_contact_state.contact_pairs.append("bin_0<->bin_1")
    env.swap_contact_state.max_force_norm = 1.5
    env.swap_contact_state.max_force_pair = "bin_0<->bin_1"
    env.swap_contact_state.max_force_step = 13
    env.swap_contact_state.pair_max_force["bin_0<->bin_1"] = 1.5

    wrapper = RobommeRecordWrapper(
        env,
        dataset=str(tmp_path),
        env_id="DummyEnv",
        episode=7,
        seed=11,
        save_video=True,
        record_hdf5=False,
    )
    wrapper.episode_success = True
    wrapper.video_frames.append(np.zeros((8, 8, 3), dtype=np.uint8))
    wrapper.close()

    videos_dir = tmp_path / "videos"
    expected_video = videos_dir / "DummyEnv_ep7_seed11_no_goal.mp4"
    assert expected_video.exists()
    assert not (videos_dir / f"swapcontact_{expected_video.name}").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
