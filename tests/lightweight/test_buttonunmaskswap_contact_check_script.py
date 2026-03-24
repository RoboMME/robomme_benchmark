import importlib.util
import json
import sys

import pytest

from tests._shared.repo_paths import find_repo_root


pytestmark = [pytest.mark.lightweight]


def _load_module(module_name: str, relative_path: str):
    repo_root = find_repo_root(__file__)
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


script_mod = _load_module(
    "button_unmask_swap_contact_script_under_test",
    "scripts/dev/buttonunmaskswap_contact_check.py",
)


def test_main_single_worker_writes_jsonl_record(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "results.jsonl"
    output_dir = tmp_path / "videos_out"

    fake_record = {
        "env": "ButtonUnmaskSwap",
        "episode": 3,
        "seed": 4,
        "difficulty": "hard",
        "episode_success": True,
        "swap_contact_detected": True,
        "first_contact_step": 71,
        "contact_pairs": ["bin_0<->bin_2"],
        "max_force_norm": 5.0,
        "max_force_pair": "bin_0<->bin_2",
        "max_force_step": 72,
        "pair_max_force": {"bin_0<->bin_2": 5.0},
        "video_path": None,
    }

    monkeypatch.setattr(script_mod, "_run_episode_worker", lambda job: dict(fake_record))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "buttonunmaskswap_contact_check.py",
            "--workers",
            "1",
            "--total-episodes",
            "1",
            "--episode-start",
            "3",
            "--seed-start",
            "4",
            "--output-dir",
            str(output_dir),
            "--jsonl-path",
            str(jsonl_path),
        ],
    )

    script_mod.main()

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload == fake_record
