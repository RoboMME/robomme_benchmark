"""Validated episode identity and oracle-derived evaluation budgets (CPU only)."""

import copy
import json
from importlib import resources
from pathlib import Path

from robomme_plus.registry import episode_budget, get_task
from robomme_plus.evaluation.timing import episode_control_frequency


def positive_integer(value, field, *, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def normalize_episode(task, record, index):
    """Copy a record, validate its identity, and recompute its budget."""
    spec = get_task(task)
    if not isinstance(record, dict):
        raise ValueError(f"Episode {index} must be a JSON object")
    result = copy.deepcopy(record)
    if task in ("ButtonUnmask", "VideoUnmask"):
        # Removal episodes belong to a retired task revision. Replaying their
        # seeds with the restored stationary task would silently change them.
        removal_fields = [key for key in result if key.startswith("remove")
                          or key == "removal_visibility_validation"]
        if any(result[key] not in (None, False) for key in removal_fields):
            raise ValueError(f"{task} removal metadata belongs to a retired task revision")
        for key in removal_fields:
            result.pop(key)
    if result.get("task", task) != task:
        raise ValueError(f"Episode {index} belongs to {result['task']!r}, not {task!r}")
    result.update(task=task, suite=spec.suite, episode=result.get("episode", index))
    positive_integer(result["episode"], "episode", allow_zero=True)
    positive_integer(result.get("seed"), "seed", allow_zero=True)
    steps = result.get("oracle_policy_step", result.get("oracle_execution_steps"))
    if steps is None:
        raise ValueError(
            f"{task} episode {index} has no oracle_policy_step in the supplied metadata"
        )
    result["oracle_policy_step"] = positive_integer(steps, "oracle_policy_step")
    result.pop("oracle_execution_steps", None)
    result["max_steps"] = episode_budget(steps)
    result["control_freq"] = episode_control_frequency(task, record)
    level = result.get("difficulty")
    if 'continuous_motion' in result:
        if task != 'ButtonUnmaskSwap' or type(result['continuous_motion']) is not bool:
            raise ValueError('continuous_motion must be a boolean for ButtonUnmaskSwap')
    if 'motion_workspace' in result:
        from robomme_plus.tasks.permanence.utils.workspace import resolve_workspace
        if task != 'ButtonUnmaskSwap' or not isinstance(result['motion_workspace'], dict):
            raise ValueError('motion_workspace requires ButtonUnmaskSwap and a workspace object')
        result['motion_workspace'] = resolve_workspace(result['motion_workspace'])
    if 'continuous_motion_version' in result:
        from robomme_plus.tasks.permanence.utils.workspace import resolve_motion_version
        if (task != 'ButtonUnmaskSwap' or not result.get('continuous_motion')
                or type(result['continuous_motion_version']) is not int):
            raise ValueError('continuous_motion_version requires ButtonUnmaskSwap continuous motion')
        result['continuous_motion_version'] = resolve_motion_version(result['continuous_motion_version'])
    if task == "InsertPeg" and level is None:
        # Historical seed-only episodes use exactly the retained hard geometry.
        level = "hard"
    if task == 'PatternLock' and 'execution_grid_scale' in result:
        from robomme_plus.tasks.imitation.utils.pattern_layout import validate_grid_scale
        result['execution_grid_scale'] = validate_grid_scale(
            result['execution_grid_scale'], level if result.get('stick_change_version') in (4, 5, 6) else None)
    if task in ('PatternLock', 'RouteStick') and 'stick_change_version' in result:
        version = result['stick_change_version']
        versions = (1, 2, 3, 4, 5, 6) if task == 'PatternLock' else (1,)
        if type(version) is not int or version not in versions:
            raise ValueError(f'{task} stick_change_version must be one of {versions}')
    if task == 'RouteStick' and any(key in result for key in ('execution_stick_angle_deg', 'execution_stick_tilt_deg')):
        raise ValueError('RouteStick only supports a straight stick; remove stick angle metadata')
    if task == 'PatternLock' and 'execution_stick_angle_deg' in result:
        if result.get('stick_change_version') not in (3, 4, 5, 6):
            raise ValueError('execution_stick_angle_deg requires stick_change_version=3, 4, 5 or 6')
        from robomme_plus.tasks.imitation.utils.variants import validate_pattern_stick_angle, validate_level_stick_angle
        result['execution_stick_angle_deg'] = (
            validate_level_stick_angle(result['execution_stick_angle_deg'], level)
            if result.get('stick_change_version') in (4, 5, 6) else validate_pattern_stick_angle(result['execution_stick_angle_deg']))
    if task == 'RouteStick' and 'execution_route_layout' in result:
        from robomme_plus.tasks.imitation.utils.route_layout import resolve_route_layout
        result['execution_route_layout'] = resolve_route_layout(result['seed'], result['execution_route_layout'])
    if task == "PickHighlight":
        from robomme_plus.tasks.reference.utils.highlight_levels import (
            PUBLIC_LEVELS,
            cube_count_range,
            native_difficulty,
            validate_color_mode,
        )
        from robomme_plus.tasks.reference.utils.highlight_timing import resolve_highlight_mode

        if level in PUBLIC_LEVELS:
            result["native_difficulty"], level = level, PUBLIC_LEVELS[level]
        result["native_difficulty"] = native_difficulty(
            level, result["seed"], result.get("native_difficulty")
        )
        # Records predating the two cue modes used staggered highlights.
        result["highlight_mode"] = resolve_highlight_mode(
            result["seed"], result.get("highlight_mode", "staggered")
        )
        result["layout_version"] = result.get("layout_version", 1)
        cube_count_range(level, result["layout_version"])
        result["local_adjustments"] = result.get("local_adjustments", False)
        result["color_mode"] = validate_color_mode(result.get("color_mode", "independent"))
        if type(result["local_adjustments"]) is not bool:
            raise ValueError("PickHighlight local_adjustments must be a boolean")
    if spec.difficulties and level not in spec.difficulties:
        raise ValueError(f"{task} difficulty must be one of {spec.difficulties}")
    if not spec.difficulties and level is not None:
        raise ValueError(f"{task} is seed-only and has no difficulty")
    result["difficulty"] = level
    from robomme_plus.variation import validate_profile
    validate_profile(task, level, result.get("perceptual_variation"))
    from robomme_plus.variation.mechanism import validate_profile as validate_mechanism
    validate_mechanism(task, level, result.get("mechanism_variation"), result.get("perceptual_variation"))
    if (isinstance(result.get('mechanism_variation'),dict)
            and result['mechanism_variation'].get('atomic')=='container_speed_2x'
            and result.get('continuous_motion')):
        raise ValueError('Atomic container_speed_2x requires a finite-swap episode')
    if result.get("mechanism_variation") is not None:
        positive_integer(result.get("base_oracle_policy_step"), "base_oracle_policy_step")
    if task == "InsertPeg" and "hole_width_m" in result:
        from robomme_plus.tasks.imitation.utils.geometry import resolve_config

        expected = 2 * resolve_config("insert_peg", result["seed"], level)["aperture_half_size_m"]
        width = result["hole_width_m"]
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not abs(width - expected) < 1e-12:
            raise ValueError("InsertPeg hole_width_m differs from the seeded difficulty geometry")
    if task == "InsertPeg" and (
        "oracle_wrist_visible_grasp" in result or "oracle_grasp_offset_m" in result
    ):
        from robomme_plus.tasks.imitation.utils.geometry import INSERT_VISIBLE_GRASP_OFFSET_M

        visible = result.get("oracle_wrist_visible_grasp", False)
        if type(visible) is not bool or (visible and level not in ("easy", "medium")):
            raise ValueError("InsertPeg oracle_wrist_visible_grasp is a boolean for easier tiers only")
        offset = result.get("oracle_grasp_offset_m")
        if visible:
            if (isinstance(offset, bool) or not isinstance(offset, (int, float))
                    or not abs(offset - INSERT_VISIBLE_GRASP_OFFSET_M) < 1e-12):
                raise ValueError("InsertPeg oracle_grasp_offset_m differs from the visible end grasp")
        elif offset is not None:
            raise ValueError("InsertPeg oracle_grasp_offset_m requires oracle_wrist_visible_grasp")
    if spec.variants:
        from robomme_plus.tasks.imitation.utils.variants import imitation_variant

        result["variant"] = imitation_variant(
            task, result["seed"], result.get("variant")
        )[0]
        if task == "MoveCube" and "scene_parameters" in result:
            parameters = result["scene_parameters"]
            if parameters.get("schema") in ("robomme-ood-move-cube/2", "robomme-ood-move-cube/3"):
                from robomme_plus.tasks.imitation.move_cube_support.config import validate_config
                validate_config(parameters, result["variant"], result["seed"])
            else:
                from robomme_plus.tasks.imitation.utils.geometry import recorded_move_config
                recorded_move_config(parameters, "move_" + result["variant"], result["seed"])

    else:
        if record.get("variant") is not None and "oracle_policy_step" in record:
            raise ValueError(f"{task} has no geometry variants")
        result.pop("variant", None)
    return result


def replay_identity(task, record):
    """Return the normalized settings consumed by the public episode runtime.

    Recording times, export paths and authoring annotations do not change an
    episode. Keep the oracle step count: it determines the execution budget.
    """
    result = normalize_episode(task, record, 0)
    fields = {
        "task", "episode", "seed", "difficulty", "oracle_policy_step", "control_freq",
        "variant", "native_difficulty", "continuous_motion", "motion_workspace",
        "continuous_motion_version", "oracle_wrist_visible_grasp", "oracle_grasp_offset_m",
        "hole_width_m", "stick_change_version", "execution_route_layout",
        "execution_stick_angle_deg", "execution_grid_scale", "scene_parameters",
        "demonstration_actions", "highlight_mode", "layout_version",
        "local_adjustments", "color_mode",
        "mechanism_variation", "base_oracle_policy_step",
    }
    if task == "InsertPeg":
        from robomme_plus.tasks.imitation.utils.geometry import resolve_config
        result["hole_width_m"] = 2 * resolve_config(
            "insert_peg", result["seed"], result["difficulty"])["aperture_half_size_m"]
    return {key: value for key, value in result.items() if key in fields}


def load_episode_metadata(task, dataset="test", *, metadata_path=None):
    """Load packaged or custom oracle records; never trust a stored max_steps."""
    get_task(task)
    if metadata_path is None:
        if dataset not in ("train", "test"):
            raise ValueError(
                "dataset must be 'train' or 'test'; use metadata_path for custom episodes"
            )
        package = __package__.split(".")[0]
        path = (
            resources.files(package)
            / "env_metadata"
            / dataset
            / f"record_dataset_{task}_metadata.json"
        )
    else:
        path = Path(metadata_path)
    document = json.loads(path.read_text())
    if isinstance(document, dict) and document.get("env_id", task) != task:
        raise ValueError(f"Metadata belongs to {document['env_id']!r}, not {task!r}")
    records = document.get("records") if isinstance(document, dict) else document
    if not isinstance(records, list) or not records:
        raise ValueError("Episode metadata must contain a nonempty records list")
    return [normalize_episode(task, row, index) for index, row in enumerate(records)]
