"""Post-test freeze supplement: policy re-verification, freeze timing and dependency change detection."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/plus_eval"))
from benchmark_io.freeze_supplement import build  # noqa: E402

FUTURE = time.time() + 3600  # files created by the test have earlier change times
PAST = 1_000_000.0           # files created by the test have later change times


def _case(root, *, first_test, dev_after_freeze=False, tamper=False):
    case = root / "case"
    policy = case / "evaluation" / "policy"
    policy.mkdir(parents=True)
    (policy / "Task.py").write_text("ACT = 1\n")
    freeze = first_test - 100
    (case / "evaluation" / "frozen_evaluation.json").write_text(json.dumps(
        {"created": freeze, "episodes": {"Task": [0]}, "policy_sha256": {"Task.py": hashlib.sha256(b"ACT = 1\n").hexdigest()}}))
    if tamper:
        (policy / "Task.py").write_text("ACT = 2\n")
    starts = {"evaluation/evaluation_attempts.sqlite3": [first_test, first_test + 9],
              "study/attempts.sqlite3": [freeze - 50] + ([freeze + 1] if dev_after_freeze else [])}
    for name, times in starts.items():
        path = case / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE attempts (id INTEGER PRIMARY KEY, task TEXT, started REAL)")
            db.executemany("INSERT INTO attempts(task, started) VALUES ('Task', ?)", [(t,) for t in times])
    return case


def _dependency(root):
    dep = root / "env"
    (dep / "lib").mkdir(parents=True)
    (dep / "lib" / "model.pt").write_bytes(b"weights")
    (dep / "python").symlink_to(dep / "lib" / "model.pt")
    return dep


def test_final_policy_and_unchanged_dependencies_pass(tmp_path):
    result = build([_case(tmp_path, first_test=FUTURE)], {"env": _dependency(tmp_path)})
    case = result["cases"][0]
    assert case["policy_matches_pre_test_manifest"] and case["development_starts_after_freeze"] == 0
    dep = result["dependencies"]["env"]
    assert dep["files"] == 2 and dep["changed_at_or_after_first_test"] == []
    link = next(r for r in dep["inventory"] if "symlink_target" in r)
    assert link["target_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert result["all_checks_pass"] is True


def test_dependency_changed_after_first_test_fails(tmp_path):
    result = build([_case(tmp_path, first_test=PAST)], {"env": _dependency(tmp_path)})
    assert len(result["dependencies"]["env"]["changed_at_or_after_first_test"]) == 2
    assert result["all_checks_pass"] is False


def test_tampered_policy_or_late_development_fails(tmp_path):
    tampered = build([_case(tmp_path / "a", first_test=FUTURE, tamper=True)], {"w": _dependency(tmp_path / "a")})
    assert tampered["cases"][0]["policy_matches_pre_test_manifest"] is False
    late = build([_case(tmp_path / "b", first_test=FUTURE, dev_after_freeze=True)], {"w": _dependency(tmp_path / "b")})
    assert late["cases"][0]["development_starts_after_freeze"] == 1
    assert tampered["all_checks_pass"] is False and late["all_checks_pass"] is False


def test_per_task_manifests_are_each_checked(tmp_path):
    """RAX-style layout: evaluation/<Task>/frozen_evaluation.json, one per task."""
    case = _case(tmp_path, first_test=FUTURE)
    per_task = case / "evaluation" / "Task"
    per_task.mkdir()
    for name in ("frozen_evaluation.json", "evaluation_attempts.sqlite3", "policy"):
        (case / "evaluation" / name).rename(per_task / name)
    result = build([case], {"env": _dependency(tmp_path)})
    assert len(result["cases"][0]["manifests"]) == 1
    assert result["cases"][0]["manifests"][0]["tasks"] == ["Task"]
    assert result["all_checks_pass"] is True


def test_pre_test_mode_writes_dependency_manifest(tmp_path):
    result = build([], {"env": _dependency(tmp_path)})
    assert result["kind"] == "pre_test_dependency_manifest" and result["cases"] == []
    assert result["dependencies"]["env"]["files"] == 2
    assert result["dependencies"]["env"]["changed_at_or_after_first_test"] == []
    assert result["all_checks_pass"] is True
