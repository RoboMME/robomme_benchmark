from pathlib import Path
from typing import Any, List, Optional

try:
    from .choice_action_mapping import select_target_with_position
except Exception:
    # Fallback for direct file loading in lightweight tests.
    import importlib.util
    import sys

    _mapping_path = Path(__file__).resolve().with_name("choice_action_mapping.py")
    _spec = importlib.util.spec_from_file_location(
        "choice_action_mapping_fallback",
        _mapping_path,
    )
    assert _spec is not None and _spec.loader is not None
    _module = importlib.util.module_from_spec(_spec)
    sys.modules.setdefault("choice_action_mapping_fallback", _module)
    _spec.loader.exec_module(_module)
    select_target_with_position = _module.select_target_with_position


def find_exact_label_option_index(target_label: Any, options: List[dict]) -> int:
    """Return option index only when target_label exactly equals option label."""
    if not isinstance(target_label, str):
        return -1
    for idx, opt in enumerate(options):
        if opt.get("label") == target_label:
            return idx
    return -1


def map_action_text_to_option_label(action_text: Any, options: List[dict]) -> Optional[str]:
    """Map exact option action text to its option label for recording-time conversion."""
    if not isinstance(action_text, str):
        return None
    for opt in options:
        if opt.get("action") == action_text:
            label = opt.get("label")
            if isinstance(label, str) and label:
                return label
            return None
    return None
