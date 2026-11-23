"""Ledger, evidence-hook and setting-name safeguards in benchmark_io; no simulator or API calls."""
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/plus_eval"))
from benchmark_io import evidence_hook  # noqa: E402
from benchmark_io.development_evidence import EvidenceRecorder  # noqa: E402
from benchmark_io.episode import Episode  # noqa: E402
from benchmark_io.protocol import BudgetExhausted, Study, setting_key  # noqa: E402

RECORDS = {"BinFill": [{"seed": 1}, {"seed": 2}, {"seed": 3}]}


def _never_build(*args, **kwargs):
    raise AssertionError("The simulator must not be constructed in these tests")


@pytest.mark.parametrize("name, expected", [
    ("Zero-shot", "zero-shot"), ("Demo-only", "demo-only"),
    ("Interaction-only", "interaction-only"), ("Hybrid", "demo&interaction"),
    ("hybrid", "demo&interaction"), ("demo&interaction", "demo&interaction"),
])
def test_setting_key_accepts_documented_names(name, expected):
    assert setting_key(name) == expected


@pytest.mark.parametrize("name", ["few-shot", "", None, 3])
def test_setting_key_rejects_unknown_names(name):
    with pytest.raises(ValueError):
        setting_key(name)


def test_refused_reservation_leaves_no_rollout_directory(tmp_path):
    study = Study(tmp_path / "study", method="m", setting="interaction-only", budget=1, records=RECORDS)
    study.finish(study.reserve("BinFill", purpose="development")["id"], {"control_steps": 0})
    output = tmp_path / "rollouts" / "refused"
    with pytest.raises(BudgetExhausted):
        Episode("BinFill", study=study, output=output, builder_factory=_never_build)
    assert not output.exists()
    assert len(study.attempts()) == 1


def test_rollout_directory_failure_finishes_the_reserved_attempt(tmp_path):
    study = Study(tmp_path / "study", method="m", setting="interaction-only", budget=2, records=RECORDS)
    output = tmp_path / "rollouts" / "taken"
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        Episode("BinFill", study=study, output=output, builder_factory=_never_build)
    (attempt,) = study.attempts()
    assert attempt["ended"] is not None
    assert "FileExistsError" in json.loads(attempt["outcome"])["error"]


@pytest.mark.parametrize("event_name, code", [
    ("PreToolUse", 2), ("Stop", 1), ("SubagentStop", 1), ("PostToolUse", 1), ("SessionEnd", 1),
])
def test_hook_blocks_only_tool_calls_when_recorder_is_down(event_name, code):
    stdin = io.StringIO(json.dumps({"hook_event_name": event_name}))
    with patch.object(sys, "stdin", stdin), \
            patch.object(evidence_hook.socket.socket, "connect", side_effect=FileNotFoundError):
        assert evidence_hook.main() == code


def test_evidence_objects_are_complete_and_content_addressed(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "policy").mkdir(parents=True)
    (workspace / "policy" / "run.py").write_text("print('hello')\n")
    recorder = EvidenceRecorder(tmp_path / "history", workspace)
    recorder.record({"hook_event_name": "PostToolUse"})
    objects = list((tmp_path / "history" / "objects").iterdir())
    assert [p.suffix for p in objects] == [""]
    import hashlib
    assert hashlib.sha256(objects[0].read_bytes()).hexdigest() == objects[0].name
