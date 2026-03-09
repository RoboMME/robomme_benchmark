from __future__ import annotations

from typing import Callable, Dict, List

from robomme.robomme_env.utils import vqa_options as upstream_vqa_options

solve_button = upstream_vqa_options.solve_button
solve_button_ready = upstream_vqa_options.solve_button_ready
solve_hold_obj_absTimestep = upstream_vqa_options.solve_hold_obj_absTimestep


def _build_stopcube_static_checkpoints(final_target: int) -> List[int]:
    checkpoints = list(range(100, final_target, 100))
    if not checkpoints or checkpoints[-1] != final_target:
        checkpoints.append(final_target)

    if len(checkpoints) >= 2 and checkpoints[-1] - checkpoints[-2] < 50:
        del checkpoints[-2]

    return checkpoints


def _options_stopcube_override(env, planner, require_target, base) -> List[dict]:
    _ = require_target
    options: List[dict] = []
    button_obj = getattr(base, "button", None)

    if button_obj is not None:
        options.append(
            {
                "label": "a",
                "action": "move to the top of the button to prepare",
                "solve": lambda button_obj=button_obj: solve_button_ready(
                    env, planner, obj=button_obj
                ),
            }
        )

    steps_press = getattr(base, "steps_press", None)
    if steps_press is not None:

        def solve_with_incremental_steps():
            steps_press_value = getattr(base, "steps_press", None)
            if steps_press_value is None:
                return None

            interval = getattr(base, "interval", 30)
            final_target = max(0, int(steps_press_value - interval))
            current_step = int(getattr(env, "elapsed_steps", 0))

            checkpoints_key = "_stopcube_static_checkpoints"
            index_key = "_stopcube_static_index"
            cached_final_target_key = "_stopcube_static_final_target"
            last_elapsed_step_key = "_stopcube_static_last_elapsed_step"

            checkpoints = getattr(base, checkpoints_key, None)
            index = getattr(base, index_key, None)
            cached_final_target = getattr(base, cached_final_target_key, None)
            last_elapsed_step = getattr(base, last_elapsed_step_key, None)

            needs_rebuild = (
                not isinstance(checkpoints, list)
                or len(checkpoints) == 0
                or index is None
                or cached_final_target is None
                or int(cached_final_target) != final_target
                or (
                    last_elapsed_step is not None
                    and current_step < int(last_elapsed_step)
                )
            )

            if needs_rebuild:
                checkpoints = _build_stopcube_static_checkpoints(final_target)
                index = 0
            else:
                index = int(index)
                if index < 0:
                    index = 0
                if index >= len(checkpoints):
                    index = len(checkpoints) - 1

            target = checkpoints[index]
            solve_hold_obj_absTimestep(env, planner, absTimestep=target)

            index += 1

            setattr(base, checkpoints_key, checkpoints)
            setattr(base, index_key, index)
            setattr(base, cached_final_target_key, final_target)
            setattr(base, last_elapsed_step_key, current_step)

            return None

        options.append(
            {
                "label": "b",
                "action": "remain static",
                "solve": solve_with_incremental_steps,
            }
        )

    if button_obj is not None:
        options.append(
            {
                "label": "c",
                "action": "press button to stop the cube",
                "solve": lambda button_obj=button_obj: solve_button(
                    env, planner, obj=button_obj, without_hold=True
                ),
            }
        )

    return options


OPTION_BUILDERS: Dict[str, Callable] = dict(upstream_vqa_options.OPTION_BUILDERS)
OPTION_BUILDERS["StopCube"] = _options_stopcube_override


def get_vqa_options(env, planner, selected_target, env_id: str) -> List[dict]:
    """Return Gradio-specific solve options without mutating the upstream src module."""

    def _require_target():
        obj = selected_target.get("obj")
        if obj is None:
            raise ValueError(
                "No available target cube found, please click target in segmentation map first."
            )
        return obj

    base = env.unwrapped
    builder = OPTION_BUILDERS.get(
        env_id, getattr(upstream_vqa_options, "_options_default")
    )
    return builder(env, planner, _require_target, base)

