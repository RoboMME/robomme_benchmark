import importlib.util
import json
import sys

import pytest

from robomme.robomme_env.utils.swap_contact_monitoring import new_swap_contact_state
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
    "contact_check_script_under_test",
    "scripts/dev/contact_check.py",
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
            "contact_check.py",
            "--env",
            "ButtonUnmaskSwap",
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


def test_maybe_prefix_video_with_swapcontact_renames_only_detected_video(tmp_path):
    video_path = tmp_path / "ButtonUnmaskSwap_ep3_seed4_demo.mp4"
    video_path.write_bytes(b"demo")

    renamed_path = script_mod._maybe_prefix_video_with_swapcontact(video_path, True)

    assert renamed_path == tmp_path / "swapcontact_ButtonUnmaskSwap_ep3_seed4_demo.mp4"
    assert renamed_path.exists()
    assert not video_path.exists()

    untouched_path = script_mod._maybe_prefix_video_with_swapcontact(renamed_path, False)
    assert untouched_path == renamed_path


def test_maybe_prefix_video_with_swapcontact_avoids_double_prefix(tmp_path):
    video_path = tmp_path / "swapcontact_ButtonUnmaskSwap_ep3_seed4_demo.mp4"
    video_path.write_bytes(b"demo")

    result_path = script_mod._maybe_prefix_video_with_swapcontact(video_path, True)

    assert result_path == video_path
    assert result_path.exists()


def test_current_swap_contact_summary_reads_shared_state():
    state = new_swap_contact_state()
    state.swap_contact_detected = True
    state.first_contact_step = 14
    state.contact_pairs.append("bin_0<->bin_1")
    state.max_force_norm = 3.5
    state.max_force_pair = "bin_0<->bin_1"
    state.max_force_step = 14
    state.pair_max_force["bin_0<->bin_1"] = 3.5

    class _Env:
        pass

    env = _Env()
    env.unwrapped = _Env()
    env.unwrapped.swap_contact_state = state

    assert script_mod._current_swap_contact_summary(env) == {
        "swap_contact_detected": True,
        "first_contact_step": 14,
        "contact_pairs": ["bin_0<->bin_1"],
        "max_force_norm": 3.5,
        "max_force_pair": "bin_0<->bin_1",
        "max_force_step": 14,
        "pair_max_force": {"bin_0<->bin_1": 3.5},
    }


def test_difficulty_for_episode_cycles_easy_medium_hard():
    assert script_mod._difficulty_for_episode(1) == "easy"
    assert script_mod._difficulty_for_episode(2) == "medium"
    assert script_mod._difficulty_for_episode(3) == "hard"
    assert script_mod._difficulty_for_episode(4) == "easy"
