"""Build choice_action JSON payloads for dataset generation."""

import json
import re
from typing import Optional

_COORD_RE = re.compile(r"<\s*(-?\d+)\s*,\s*(-?\d+)\s*>")


def build_choice_dict_find_point(
    env_id: Optional[str],
    vqa_label: Optional[str],
    grounded_subgoal: Optional[str],
) -> str:
    """
    Build serialized choice_action for dataset recording.

    Current scope is PickXtimes only:
    - non-PickXtimes returns "{}"
    - missing vqa_label returns "{}"
    - point is extracted from the first "<y, x>" in grounded_subgoal
    """
    if env_id != "PickXtimes":
        return "{}"

    if not isinstance(vqa_label, str) or not vqa_label.strip():
        return "{}"

    payload = {"choice": vqa_label}
    if grounded_subgoal:
        matched = _COORD_RE.search(str(grounded_subgoal))
        if matched is not None:
            payload["point"] = [int(matched.group(1)), int(matched.group(2))]

    return json.dumps(payload)