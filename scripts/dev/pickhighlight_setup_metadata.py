from __future__ import annotations

import json
from pathlib import Path

import h5py

PICKHIGHLIGHT_ENV_ID = "PickHighlight"
PICKHIGHLIGHT_METADATA_DATASET = "pickhighlight_metadata"
VALID_TARGET_COLORS = {"red", "blue", "green"}


def _resolve_env_id(env) -> str:
    base_env = getattr(env, "unwrapped", env)
    spec = getattr(base_env, "spec", None)
    spec_id = getattr(spec, "id", None)
    if spec_id:
        return str(spec_id)
    env_id = getattr(env, "env_id", None)
    return str(env_id or "")


def _extract_target_cube_colors(base_env) -> list[str]:
    raw_colors = list(getattr(base_env, "target_cube_colors", []) or [])
    if not raw_colors:
        raise ValueError("PickHighlight env missing target_cube_colors")

    target_colors: list[str] = []
    for color in raw_colors:
        color_name = str(color).strip().lower()
        if color_name not in VALID_TARGET_COLORS:
            raise ValueError(f"invalid PickHighlight target cube color: {color!r}")
        target_colors.append(color_name)
    return target_colors


def write_pickhighlight_setup_metadata(env, h5_path: Path, episode: int) -> None:
    """Append ordered PickHighlight target colors into the setup group."""
    if _resolve_env_id(env) != PICKHIGHLIGHT_ENV_ID:
        return

    base_env = getattr(env, "unwrapped", env)
    payload = {
        "target_cube_colors": _extract_target_cube_colors(base_env),
    }

    episode_group_name = f"episode_{episode}"
    with h5py.File(h5_path, "a") as handle:
        episode_group = handle.get(episode_group_name)
        if not isinstance(episode_group, h5py.Group):
            raise KeyError(f"missing group '{episode_group_name}'")

        setup_group = episode_group.get("setup")
        if not isinstance(setup_group, h5py.Group):
            raise KeyError(f"{episode_group_name} missing setup group")

        if PICKHIGHLIGHT_METADATA_DATASET in setup_group:
            del setup_group[PICKHIGHLIGHT_METADATA_DATASET]

        setup_group.create_dataset(
            PICKHIGHLIGHT_METADATA_DATASET,
            data=json.dumps(payload, ensure_ascii=True),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
