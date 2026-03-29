from __future__ import annotations

import json
import re
from pathlib import Path

import h5py

VIDEOREPICK_ENV_ID = "VideoRepick"
VIDEOREPICK_METADATA_DATASET = "videorepick_metadata"
COLOR_TOLERANCE = 0.05
COLOR_RGB_MAP = {
    "red": (1.0, 0.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0),
}
COLOR_NAME_PATTERN = re.compile(r"\b(red|blue|green)\b")


def _resolve_env_id(env) -> str:
    base_env = getattr(env, "unwrapped", env)
    spec = getattr(base_env, "spec", None)
    spec_id = getattr(spec, "id", None)
    if spec_id:
        return str(spec_id)
    env_id = getattr(env, "env_id", None)
    return str(env_id or "")


def _normalize_rgba(base_color) -> tuple[float, float, float, float] | None:
    try:
        color_values = [float(value) for value in list(base_color)]
    except Exception:
        return None

    if len(color_values) == 3:
        color_values.append(1.0)
    if len(color_values) < 4:
        return None
    return (
        color_values[0],
        color_values[1],
        color_values[2],
        color_values[3],
    )


def _match_rgb_to_name(rgba: tuple[float, float, float, float]) -> str | None:
    for color_name, expected_rgb in COLOR_RGB_MAP.items():
        if all(
            abs(rgba[channel_idx] - expected_rgb[channel_idx]) <= COLOR_TOLERANCE
            for channel_idx in range(3)
        ):
            return color_name
    return None


def _iter_actor_rgba_colors(actor):
    for entity in list(getattr(actor, "_objs", []) or []):
        for component in list(getattr(entity, "components", []) or []):
            render_shapes = getattr(component, "render_shapes", None)
            if render_shapes is None:
                continue
            for render_shape in list(render_shapes):
                material = getattr(render_shape, "material", None)
                if material is None:
                    get_material = getattr(render_shape, "get_material", None)
                    if callable(get_material):
                        try:
                            material = get_material()
                        except Exception:
                            material = None
                if material is None:
                    continue

                base_color = getattr(material, "base_color", None)
                if base_color is None:
                    get_base_color = getattr(material, "get_base_color", None)
                    if callable(get_base_color):
                        try:
                            base_color = get_base_color()
                        except Exception:
                            base_color = None
                if base_color is None:
                    continue

                rgba = _normalize_rgba(base_color)
                if rgba is not None:
                    yield rgba


def _color_from_actor_name(actor_name: object) -> str | None:
    if not actor_name:
        return None
    match = COLOR_NAME_PATTERN.search(str(actor_name).lower())
    if match is None:
        return None
    return match.group(1)


def _extract_target_cube_color(target_cube) -> str:
    for rgba in _iter_actor_rgba_colors(target_cube):
        color_name = _match_rgb_to_name(rgba)
        if color_name is not None:
            return color_name

    actor_name = getattr(target_cube, "name", None)
    color_name = _color_from_actor_name(actor_name)
    if color_name is not None:
        return color_name

    raise ValueError(
        "failed to resolve target_cube_1_color from render material or actor name"
    )


def _extract_num_repeats(base_env) -> int:
    num_repeats = getattr(base_env, "num_repeats", None)
    if isinstance(num_repeats, bool):
        raise ValueError("num_repeats must be a positive integer, got bool")
    try:
        repeat_count = int(num_repeats)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid num_repeats: {num_repeats!r}") from exc
    if repeat_count < 1:
        raise ValueError(f"num_repeats must be >= 1, got {repeat_count}")
    return repeat_count


def write_videorepick_setup_metadata(env, h5_path: Path, episode: int) -> None:
    """Append structured VideoRepick metadata into the setup group."""
    if _resolve_env_id(env) != VIDEOREPICK_ENV_ID:
        return

    base_env = getattr(env, "unwrapped", env)
    target_cube = getattr(base_env, "target_cube_1", None)
    if target_cube is None:
        raise ValueError("VideoRepick env missing target_cube_1")

    payload = {
        "target_cube_1_color": _extract_target_cube_color(target_cube),
        "num_repeats": _extract_num_repeats(base_env),
    }

    episode_group_name = f"episode_{episode}"
    with h5py.File(h5_path, "a") as handle:
        episode_group = handle.get(episode_group_name)
        if not isinstance(episode_group, h5py.Group):
            raise KeyError(f"missing group '{episode_group_name}'")

        setup_group = episode_group.get("setup")
        if not isinstance(setup_group, h5py.Group):
            raise KeyError(f"{episode_group_name} missing setup group")

        if VIDEOREPICK_METADATA_DATASET in setup_group:
            del setup_group[VIDEOREPICK_METADATA_DATASET]

        setup_group.create_dataset(
            VIDEOREPICK_METADATA_DATASET,
            data=json.dumps(payload, ensure_ascii=True),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
