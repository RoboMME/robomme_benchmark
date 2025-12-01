"""Utility helpers for validating and normalizing HistoryBench difficulty hints."""

from __future__ import annotations

from typing import Optional


VALID_HISTORYBENCH_DIFFICULTIES = {"easy", "medium", "hard"}


def normalize_historybench_difficulty(value: Optional[str]) -> Optional[str]:
    """Return a canonical difficulty string or ``None`` if no override was provided."""
    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(
            "HistoryBench_difficulty must be a string (got "
            f"{type(value).__name__!r})."
        )

    normalized = value.strip().lower()
    if normalized not in VALID_HISTORYBENCH_DIFFICULTIES:
        raise ValueError(
            "Unsupported difficulty level. Available options: "
            f"{sorted(VALID_HISTORYBENCH_DIFFICULTIES)}."
        )

    return normalized
