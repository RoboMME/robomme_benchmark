"""ButtonUnmask: staged reveals and ordered container pickup."""

from mani_skill.utils.registration import register_env

from robomme_plus.registry import TASKS
from robomme_plus.tasks.permanence.utils.environment import PermanenceTask


@register_env(TASKS["ButtonUnmask"].gym_id)
class ButtonUnmask(PermanenceTask):
    PERCEPTUAL_VARIATIONS = True
    task_name = "ButtonUnmask"
