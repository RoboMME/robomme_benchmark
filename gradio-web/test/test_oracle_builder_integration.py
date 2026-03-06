from __future__ import annotations

from pathlib import Path


class _DummyPlanner:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _FakeRobot:
    def __init__(self):
        self.pose = object()


class _FakeAgent:
    def __init__(self):
        self.robot = _FakeRobot()


class _FakeUnwrapped:
    def __init__(self):
        self.agent = _FakeAgent()
        self.segmentation_id_map = {}

    def evaluate(self, solve_complete_eval=False):
        return {"success": False, "fail": False}


class _FakeEnv:
    def __init__(self):
        self.unwrapped = _FakeUnwrapped()
        self.demonstration_data = {"language goal": "test goal", "frames": ["f1", "f2"]}
        self.non_demonstration_task_length = 7
        self.frames = []
        self.wrist_frames = []
        self.closed = False

    def reset(self):
        return None

    def close(self):
        self.closed = True


class _FakeEnvTupleDemo(_FakeEnv):
    def __init__(self):
        super().__init__()
        self.demonstration_data = (
            {"front_rgb_list": ["tuple_f1", "tuple_f2"]},
            None,
            None,
            None,
            {"task_goal": ["tuple goal", "backup goal"]},
        )


class _BuilderSuccess:
    last_init_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = kwargs

    def get_episode_num(self):
        return 3

    def resolve_episode(self, episode_idx):
        return 123, "hard"

    def make_env_for_episode(self, episode_idx):
        return _FakeEnv()


class _BuilderTupleDemo(_BuilderSuccess):
    def make_env_for_episode(self, episode_idx):
        return _FakeEnvTupleDemo()


class _BuilderNoMetadata:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_episode_num(self):
        return 0


class _BuilderRaiseOnMake:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_episode_num(self):
        return 1

    def resolve_episode(self, episode_idx):
        return None, None

    def make_env_for_episode(self, episode_idx):
        raise RuntimeError("boom")


def test_load_episode_uses_benchmark_builder(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    monkeypatch.setenv("ROBOMME_METADATA_ROOT", "/tmp/meta-root")
    monkeypatch.setattr(oracle_logic, "BenchmarkEnvBuilder", _BuilderSuccess)
    monkeypatch.setattr(oracle_logic, "FailAwarePandaArmMotionPlanningSolver", _DummyPlanner)
    monkeypatch.setattr(oracle_logic, "FailAwarePandaStickMotionPlanningSolver", _DummyPlanner)
    monkeypatch.setattr(oracle_logic.OracleSession, "update_observation", lambda self: ("IMG", "Ready"))

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    img, msg = session.load_episode("BinFill", 1)

    assert img == "IMG"
    assert msg == "Ready"
    assert session.env_id == "BinFill"
    assert session.episode_idx == 1
    assert session.seed == 123
    assert session.difficulty == "hard"
    assert session.language_goal == "test goal"
    assert session.demonstration_frames == ["f1", "f2"]

    init_kwargs = _BuilderSuccess.last_init_kwargs
    assert init_kwargs["dataset"] == "train"
    assert init_kwargs["action_space"] == "joint_angle"
    assert init_kwargs["gui_render"] is False
    assert init_kwargs["max_steps"] == 3000
    assert init_kwargs["override_metadata_path"] == Path("/tmp/meta-root")


def test_load_episode_metadata_missing_returns_stable_error(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    monkeypatch.setenv("ROBOMME_METADATA_ROOT", "/tmp/custom-metadata")
    monkeypatch.setattr(oracle_logic, "BenchmarkEnvBuilder", _BuilderNoMetadata)

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    img, msg = session.load_episode("RouteStick", 0)

    assert img is None
    assert "Dataset metadata not found or empty" in msg
    assert "record_dataset_RouteStick_metadata.json" in msg


def test_load_episode_out_of_range_returns_stable_error(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    monkeypatch.setattr(oracle_logic, "BenchmarkEnvBuilder", _BuilderSuccess)

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    img, msg = session.load_episode("BinFill", 99)

    assert img is None
    assert "Episode index out of range" in msg
    assert "valid 0-2" in msg


def test_load_episode_init_failure_is_caught(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    monkeypatch.setattr(oracle_logic, "BenchmarkEnvBuilder", _BuilderRaiseOnMake)

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    img, msg = session.load_episode("BinFill", 0)

    assert img is None
    assert msg.startswith("Error initializing episode:")


def test_load_episode_supports_tuple_demonstration_data(monkeypatch, reload_module):
    oracle_logic = reload_module("oracle_logic")

    monkeypatch.setattr(oracle_logic, "BenchmarkEnvBuilder", _BuilderTupleDemo)
    monkeypatch.setattr(oracle_logic, "FailAwarePandaArmMotionPlanningSolver", _DummyPlanner)
    monkeypatch.setattr(oracle_logic, "FailAwarePandaStickMotionPlanningSolver", _DummyPlanner)
    monkeypatch.setattr(oracle_logic.OracleSession, "update_observation", lambda self: ("IMG", "Ready"))

    session = oracle_logic.OracleSession(dataset_root=None, gui_render=False)
    img, msg = session.load_episode("BinFill", 0)

    assert img == "IMG"
    assert msg == "Ready"
    assert session.language_goal == "backup goal"
    assert session.demonstration_frames == ["tuple_f1", "tuple_f2"]
