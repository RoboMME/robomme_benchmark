from pathlib import Path

import pytest

from tests._shared.repo_paths import find_repo_root


pytestmark = [pytest.mark.lightweight]


ENV_FILES = [
    "src/robomme/robomme_env/VideoUnmaskSwap.py",
    "src/robomme/robomme_env/ButtonUnmaskSwap.py",
    "src/robomme/robomme_env/VideoRepick.py",
    "src/robomme/robomme_env/VideoPlaceButton.py",
    "src/robomme/robomme_env/VideoPlaceOrder.py",
]


def _source(relative_path: str) -> str:
    repo_root = find_repo_root(__file__)
    return (repo_root / relative_path).read_text(encoding="utf-8")


def test_swap_envs_import_shared_selector():
    for relative_path in ENV_FILES:
        source = _source(relative_path)
        assert "from .utils.swap_selection import select_dynamic_swap_pair" in source


def test_swap_contact_monitoring_envs_import_shared_monitoring():
    for relative_path in ENV_FILES[:3]:
        source = _source(relative_path)
        assert "from .utils import swap_contact_monitoring as swapContact" in source
        assert "swapContact.detect_swap_contacts(" in source
        assert "swap_contact_state" in source


def test_swap_envs_no_longer_define_local_selection_helpers():
    for relative_path in ENV_FILES:
        source = _source(relative_path)
        assert "_compute_dynamic_swap_candidates" not in source
        assert "_select_swap_pair_from_positions" not in source


def test_button_unmask_swap_no_longer_defines_local_contact_monitoring_helpers():
    source = _source("src/robomme/robomme_env/ButtonUnmaskSwap.py")
    assert "def _reset_swap_contact_monitoring" not in source
    assert "def _detect_swap_bin_contacts" not in source
    assert "def get_swap_contact_summary" not in source


def test_video_place_order_uses_singular_specialflag_name():
    source = _source("src/robomme/robomme_env/VideoPlaceOrder.py")
    assert "current_task_specialflags" not in source
    assert "current_task_specialflag" in source
