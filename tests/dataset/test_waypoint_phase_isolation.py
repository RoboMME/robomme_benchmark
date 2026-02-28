from __future__ import annotations

import h5py
import numpy as np
import pytest

from tests._shared.dataset_generation import DatasetCase

pytestmark = pytest.mark.dataset


def _make_case(env_id: str, base_seed: int) -> DatasetCase:
    return DatasetCase(
        env_id=env_id,
        episode=0,
        base_seed=base_seed,
        difficulty="easy",
        save_video=True,
        mode_tag="waypoint_phase_isolation",
    )


def _decode_h5_string(raw) -> str:
    if isinstance(raw, np.ndarray):
        raw = raw.flatten()[0]
    if isinstance(raw, (bytes, np.bytes_)):
        return raw.decode("utf-8")
    return str(raw)


def _collect_records(ep_group: h5py.Group) -> list[dict]:
    timestep_keys = sorted(
        (k for k in ep_group.keys() if k.startswith("timestep_")),
        key=lambda k: int(k.split("_")[1]),
    )
    out: list[dict] = []
    for key in timestep_keys:
        ts = ep_group[key]
        info = ts["info"]
        waypoint_action = np.asarray(ts["action"]["waypoint_action"][()]).flatten()
        out.append(
            {
                "timestep": int(key.split("_")[1]),
                "is_demo": bool(np.reshape(np.asarray(info["is_video_demo"][()]), -1)[0]),
                "subgoal": _decode_h5_string(info["simple_subgoal_online"][()]),
                "waypoint_action": waypoint_action,
                "is_finite_waypoint": bool(
                    waypoint_action.shape == (7,) and np.all(np.isfinite(waypoint_action))
                ),
            }
        )
    return out


def _find_demo_to_non_demo_boundary(records: list[dict]) -> int | None:
    for idx in range(1, len(records)):
        if records[idx - 1]["is_demo"] and not records[idx]["is_demo"]:
            return idx
    return None


def _last_finite_demo_waypoint(records: list[dict], boundary_idx: int) -> np.ndarray | None:
    for idx in range(boundary_idx - 1, -1, -1):
        row = records[idx]
        if not row["is_demo"]:
            continue
        if row["is_finite_waypoint"]:
            return np.asarray(row["waypoint_action"]).flatten()
    return None


def _unique_finite_waypoints(rows: list[dict]) -> list[np.ndarray]:
    uniques: list[np.ndarray] = []
    prev: np.ndarray | None = None
    for row in rows:
        if not row["is_finite_waypoint"]:
            continue
        wa = np.asarray(row["waypoint_action"]).flatten()
        if prev is None or not np.array_equal(wa, prev):
            uniques.append(wa.copy())
            prev = wa.copy()
    return uniques


@pytest.mark.parametrize(
    "env_id,base_seed,assert_first_segment_midpoint",
    [
        ("PatternLock", 15001, False),
        ("RouteStick", 16000, True),
    ],
)
def test_waypoint_isolation_across_demo_phase(
    env_id: str,
    base_seed: int,
    assert_first_segment_midpoint: bool,
    dataset_factory,
):
    generated = dataset_factory(_make_case(env_id, base_seed))

    with h5py.File(generated.raw_h5_path, "r") as h5f:
        records = _collect_records(h5f["episode_0"])

    assert records, f"{env_id}: episode_0 has no recorded timesteps."
    boundary_idx = _find_demo_to_non_demo_boundary(records)
    assert boundary_idx is not None, f"{env_id}: missing demo->non-demo boundary."

    first_non_demo = records[boundary_idx]
    assert not first_non_demo["is_demo"], (
        f"{env_id}: boundary row should be non-demo, got demo at timestep "
        f"{first_non_demo['timestep']}."
    )

    last_demo_waypoint = _last_finite_demo_waypoint(records, boundary_idx)
    if first_non_demo["is_finite_waypoint"] and last_demo_waypoint is not None:
        assert not np.array_equal(
            np.asarray(first_non_demo["waypoint_action"]).flatten(),
            last_demo_waypoint,
        ), (
            f"{env_id}: boundary non-demo step consumed demo-phase pending waypoint "
            f"(timestep {first_non_demo['timestep']})."
        )

    non_demo_rows = [row for row in records if not row["is_demo"]]
    assert non_demo_rows, f"{env_id}: no non-demo rows."
    all_non_demo_unique = _unique_finite_waypoints(non_demo_rows)
    assert all_non_demo_unique, f"{env_id}: non-demo phase has no finite waypoints."

    if assert_first_segment_midpoint:
        first_subgoal = non_demo_rows[0]["subgoal"]
        first_segment_rows: list[dict] = []
        for row in non_demo_rows:
            if row["subgoal"] != first_subgoal:
                break
            first_segment_rows.append(row)
        first_segment_unique = _unique_finite_waypoints(first_segment_rows)
        assert len(first_segment_unique) >= 2, (
            f"{env_id}: first non-demo subgoal segment lost midpoint waypoint; "
            f"expected >=2 unique finite waypoints, got {len(first_segment_unique)}."
        )
