"""Fail-aware wrappers for ManiSkill motion planners."""
from __future__ import annotations

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)


class ScrewPlanFailure(RuntimeError):
    """Raised when mplib reports a screw-planning failure."""


class _FailAwareMixin:
    """Mixin that turns ``-1`` screw-plan return values into exceptions."""

    def move_to_pose_with_screw(self, *args, **kwargs):  # type: ignore[override]
        result = super().move_to_pose_with_screw(*args, **kwargs)  # type: ignore[misc]
        if isinstance(result, int) and result == -1:
            raise ScrewPlanFailure("screw plan failed")
        return result


class FailAwarePandaArmMotionPlanningSolver(_FailAwareMixin, PandaArmMotionPlanningSolver):
    """Panda arm solver that raises on screw failures."""


class FailAwarePandaStickMotionPlanningSolver(
    _FailAwareMixin, PandaStickMotionPlanningSolver
):
    """Stick solver variant that raises on screw failures."""
