"""Accounting and integration checks with synthetic environments; no rollouts."""
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from types import SimpleNamespace

import pytest

from benchmark_io.protocol import BudgetExhausted, Study, canonical
from benchmark_io.training import TrainingSession


def records():
    return {task: [dict(seed=100+i, max_steps=20, control_freq=20) for i in range(n)]
            for task, n in (("BinFill", 3), ("MoveCube", 5))}


def study(path, setting="interaction-only", budget=10):
    return Study(path, method="my_agent", setting=setting, budget=budget, records=records())


def test_explicit_seed_does_not_advance_auto_rotation_and_resume_keeps_cursor(tmp_path):
    s = study(tmp_path)
    assert s.reserve("BinFill", seed=102, purpose="verification")["episode"] == 2
    assert s.reserve("BinFill", purpose="explore")["episode"] == 0
    assert Study.open(tmp_path).reserve("BinFill", purpose="explore")["episode"] == 1
    assert s.reserve("BinFill", 0, purpose="revisit")["episode"] == 0
    assert s.reserve("BinFill", purpose="explore")["episode"] == 2
    assert s.reserve("BinFill", purpose="explore")["episode"] == 0
    assert [s.reserve("MoveCube", purpose="explore")["episode"] for _ in range(6)] == [0,1,2,3,4,0]


def test_concurrent_workers_share_total_even_when_always_choosing_one_seed(tmp_path):
    study(tmp_path)
    def reserve(_):
        try:
            return Study.open(tmp_path).reserve("BinFill", seed=102, purpose="verify", policy_revision="sha256:example")
        except BudgetExhausted:
            return None
    with ThreadPoolExecutor(max_workers=12) as pool:
        accepted = [a for a in pool.map(reserve, range(30)) if a]
    assert len(accepted) == 10
    assert sorted(a["ordinal"] for a in accepted) == list(range(1,11))
    assert {a["seed"] for a in accepted} == {102}
    s = Study.open(tmp_path)
    s.finish(accepted[0]["id"], {"error": "failed setup", "control_steps": 0})
    with pytest.raises(BudgetExhausted):
        s.reserve("BinFill", purpose="retry")
    with sqlite3.connect(s.path) as db:
        assert db.execute("SELECT count(*) FROM attempt_provenance WHERE selection='seed' AND policy_revision=?",
                          ("sha256:example",)).fetchone()[0] == 10


@pytest.mark.parametrize("setting", ["demo-only", "zero-shot"])
def test_no_interaction_setting_rejects_before_builder(tmp_path, monkeypatch, setting):
    s = study(tmp_path, setting, 0)
    def forbidden(*args, **kwargs):
        raise AssertionError("Must reject before constructing a simulator")
    monkeypatch.setitem(__import__('sys').modules, "robomme_plus", SimpleNamespace(BenchmarkEnvBuilder=forbidden))
    with TrainingSession(s.root, "BinFill") as session:
        with pytest.raises(BudgetExhausted):
            session.reset(policy_revision="commit:1")
    assert not s.attempts()
    if setting == "zero-shot":
        with pytest.raises(PermissionError):
            s.require_demos()
    else:
        s.require_demos()


def test_unknown_seed_and_changed_config_cannot_reset_budget(tmp_path):
    s = study(tmp_path)
    for kwargs in ({"seed": 999}, {"seed": True}, {"seed": 100, "episode": 0}):
        with pytest.raises(ValueError):
            s.reserve("BinFill", purpose="invalid", **kwargs)
    assert s.attempts() == []
    s.reserve("BinFill", purpose="test")
    with pytest.raises(ValueError, match="configuration"):
        study(tmp_path, budget=5)
    assert len(Study.open(tmp_path).attempts()) == 1


def test_old_study_preserves_pinned_selection_and_config(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    config = dict(method="rax", setting="interaction-only", budget=10, records=records())
    with sqlite3.connect(tmp_path/"attempts.sqlite3") as db:
        db.execute("CREATE TABLE config(id INTEGER PRIMARY KEY, json TEXT)")
        db.execute("INSERT INTO config VALUES (1,?)", (canonical(config),))
    s = Study.open(tmp_path)
    assert s.config == config
    s.reserve("BinFill", 0, purpose="explicit")
    assert s.reserve("BinFill", purpose="legacy_balanced")["episode"] == 1
    with sqlite3.connect(s.path) as db:
        assert json.loads(db.execute("SELECT json FROM config").fetchone()[0]) == config


def test_session_lifecycle_records_early_close_and_failed_construction(tmp_path, monkeypatch):
    s = study(tmp_path)
    from benchmark_io.episode import Episode
    class BrokenBuilder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("setup failure")
    original = Episode
    def broken(*args, **kwargs):
        return original(*args, **kwargs, builder_factory=BrokenBuilder)
    monkeypatch.setattr("benchmark_io.episode.Episode", broken)
    with TrainingSession(s.root, "BinFill") as session:
        with pytest.raises(RuntimeError, match="setup failure"):
            session.reset(seed=100, policy_revision="commit:before-reset")
        assert session.status()["used"] == 1
        with pytest.raises(RuntimeError, match="No open"):
            session.observe()
    a = s.attempts()[0]
    assert a["ended"] is not None and "setup failure" in json.loads(a["outcome"])["error"]
