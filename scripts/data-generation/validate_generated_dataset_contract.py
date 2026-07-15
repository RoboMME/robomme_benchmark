#!/usr/bin/env python3
"""Standalone validation of the HDF5 and metadata contract for No-Patch generated data."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_ROOT = REPO_ROOT / "src" / "robomme" / "env_metadata" / "train"
REFERENCE_ROOT = REPO_ROOT / "data" / "robomme_data_h5"
ALL_TASKS = (
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "PickHighlight",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
)
MAX_EPISODES = 100
MAX_ERRORS = 200
EPISODE_RE = re.compile(r"^episode_(\d+)$")
TIMESTEP_RE = re.compile(r"^timestep_(\d+)$")


class DatasetContractError(RuntimeError):
    """Generated data, reference data, or train metadata violates the fixed contract."""


def add_error(section: dict[str, Any], message: str) -> None:
    """Accumulate a bounded number of readable errors to prevent corrupt data from producing an unbounded report."""
    section["error_count"] = int(section.get("error_count", 0)) + 1
    errors = section.setdefault("errors", [])
    if len(errors) < MAX_ERRORS:
        errors.append(message)


def _integer(value: Any, field: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DatasetContractError(f"{path}: {field} must be an integer; got {value!r}")
    return int(value)


def read_train_metadata(
    metadata_root: str | Path = METADATA_ROOT,
    tasks: Sequence[str] = ALL_TASKS,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Strictly load train metadata from the current branch without silent fallback."""
    root = Path(metadata_root).expanduser().resolve()
    ordered_tasks = tuple(tasks)
    if not ordered_tasks or len(ordered_tasks) != len(set(ordered_tasks)):
        raise DatasetContractError("metadata task set must be non-empty and unique")
    expected_files = {f"record_dataset_{task}_metadata.json" for task in ordered_tasks}
    actual_files = {path.name for path in root.glob("record_dataset_*_metadata.json")}
    if actual_files != expected_files:
        raise DatasetContractError(
            "train metadata file set does not match the fixed task scope: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )

    all_records: dict[str, dict[int, dict[str, Any]]] = {}
    for task in ordered_tasks:
        path = root / f"record_dataset_{task}_metadata.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetContractError(f"unable to read {path}: {exc}") from exc
        if not isinstance(payload, Mapping) or payload.get("env_id") != task:
            raise DatasetContractError(f"{path}: env_id mismatch")
        records = payload.get("records")
        if not isinstance(records, list):
            raise DatasetContractError(f"{path}: records must be a list")
        if _integer(payload.get("record_count"), "record_count", path) != len(records):
            raise DatasetContractError(f"{path}: record_count does not equal the length of records")
        if len(records) != 100:
            raise DatasetContractError(f"{path}: records must contain exactly 100 entries")

        indexed: dict[int, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise DatasetContractError(f"{path}: record must be an object")
            if any(key not in record for key in ("task", "episode", "seed", "difficulty")):
                raise DatasetContractError(f"{path}: record is missing task/episode/seed/difficulty")
            if record["task"] != task:
                raise DatasetContractError(f"{path}: record task mismatch")
            episode = _integer(record["episode"], "episode", path)
            seed = _integer(record["seed"], "seed", path)
            difficulty = record["difficulty"]
            if not isinstance(difficulty, str) or not difficulty:
                raise DatasetContractError(f"{path}: difficulty must be a non-empty string")
            if episode in indexed:
                raise DatasetContractError(f"{path}: episode {episode} is duplicated")
            indexed[episode] = {
                "task": task,
                "episode": episode,
                "seed": seed,
                "difficulty": difficulty,
            }
        if set(indexed) != set(range(100)):
            raise DatasetContractError(f"{path}: episode set must exactly be 0..99")
        all_records[task] = indexed
    return all_records


def parse_tasks(value: str) -> list[str]:
    """Parse all or comma-separated task names into the task list in canonical order."""
    if value.strip().lower() == "all":
        return list(ALL_TASKS)
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names or len(names) != len(set(names)):
        raise DatasetContractError("--env must be non-empty and unique")
    unknown = sorted(set(names) - set(ALL_TASKS))
    if unknown:
        raise DatasetContractError("unknown environment: " + ", ".join(unknown))
    return [task for task in ALL_TASKS if task in names]


def episode_groups(
    handle: h5py.File,
    label: str,
    section: dict[str, Any],
) -> dict[int, h5py.Group]:
    """Return valid episode groups and record unexpected root objects as contract errors."""
    groups: dict[int, h5py.Group] = {}
    for name in handle.keys():
        match = EPISODE_RE.fullmatch(name)
        if match is None or not isinstance(handle[name], h5py.Group):
            add_error(section, f"{label}: invalid root object {name!r}")
            continue
        groups[int(match.group(1))] = handle[name]
    return groups


def timestep_indices(group: h5py.Group, source: str) -> tuple[list[int], list[str]]:
    """Parse numeric timestep names and strictly validate contiguity; valid setup groups do not count as timesteps."""
    errors: list[str] = []
    indices: list[int] = []
    for name in group.keys():
        if name == "setup":
            continue
        match = TIMESTEP_RE.fullmatch(name)
        if match is None or not isinstance(group[name], h5py.Group):
            errors.append(f"{source}: invalid timestep {name!r}")
        else:
            indices.append(int(match.group(1)))
    indices.sort()
    if not indices:
        errors.append(f"{source}: no timesteps")
    elif indices != list(range(len(indices))):
        errors.append(f"{source}: timesteps must be contiguous from 0; got {indices[:12]}")
    return indices, errors


def inspect_episode_terminal(
    group: h5py.Group,
    source: str,
) -> tuple[list[int], bool | None, list[str]]:
    """Read the strict boolean scalar info/is_completed from the final numeric timestep."""
    indices, errors = timestep_indices(group, source)
    if errors:
        return indices, None, errors
    try:
        dataset = group[f"timestep_{indices[-1]}"]["info"]["is_completed"]
    except KeyError:
        return indices, None, [f"{source}: final timestep is missing info/is_completed"]
    if (
        not isinstance(dataset, h5py.Dataset)
        or dataset.shape != ()
        or np.dtype(dataset.dtype) != np.dtype(bool)
    ):
        return indices, None, [f"{source}: info/is_completed must be a bool scalar"]
    value = dataset[()]
    if not isinstance(value, (bool, np.bool_)):
        return indices, None, [f"{source}: info/is_completed is not a bool"]
    return indices, bool(value), []


def _text(dataset: h5py.Dataset, source: str) -> str:
    if dataset.shape != ():
        raise DatasetContractError(f"{source}: string must be a scalar")
    value = dataset.asstr()[()]
    if not isinstance(value, str):
        raise DatasetContractError(f"{source}: is not a string")
    return value


def _audit_episode(
    group: h5py.Group,
    task: str,
    episode: int,
    record: Mapping[str, Any],
    label: str,
    section: dict[str, Any],
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "episode": episode,
        "timestep_count": 0,
        "final_is_completed": None,
        "joint_shape": None,
        "joint_dtype": None,
    }
    setup = group.get("setup")
    if not isinstance(setup, h5py.Group):
        add_error(section, f"{label}: missing setup")
    else:
        seed = setup.get("seed")
        difficulty = setup.get("difficulty")
        if not isinstance(seed, h5py.Dataset) or seed.shape != ():
            add_error(section, f"{label}: missing or invalid setup/seed")
        else:
            try:
                if int(seed[()]) != int(record["seed"]):
                    add_error(section, f"{label}: setup/seed mismatch")
            except (TypeError, ValueError, OverflowError):
                add_error(section, f"{label}: setup/seed is not a comparable integer")
        if not isinstance(difficulty, h5py.Dataset):
            add_error(section, f"{label}: missing setup/difficulty")
        else:
            try:
                if _text(difficulty, f"{label}: setup/difficulty") != record["difficulty"]:
                    add_error(section, f"{label}: setup/difficulty mismatch")
            except DatasetContractError as exc:
                add_error(section, str(exc))

    steps, done, terminal_errors = inspect_episode_terminal(group, label)
    for error in terminal_errors:
        add_error(section, error)
    if terminal_errors:
        return detail
    detail["timestep_count"] = len(steps)
    detail["final_is_completed"] = done

    signatures: set[tuple[tuple[int, ...], str]] = set()
    for timestep in steps:
        try:
            joint = group[f"timestep_{timestep}"]["action"]["joint_action"]
        except KeyError:
            add_error(section, f"{label}: timestep_{timestep} missing action/joint_action")
            continue
        if not isinstance(joint, h5py.Dataset):
            add_error(section, f"{label}: joint_action must be a dataset")
            continue
        signature = (tuple(joint.shape), str(joint.dtype))
        signatures.add(signature)
        if tuple(joint.shape) != (8,) or np.dtype(joint.dtype) != np.dtype(np.float64):
            add_error(section, f"{label}: joint_action must be (8,) float64")
            continue
        values = np.asarray(joint[()])
        if not np.all(np.isfinite(values)):
            add_error(section, f"{label}: joint_action contains non-finite values")
            continue
        section["joint_vector_count"] += 1
        section["joint_element_count"] += int(values.size)
    if len(signatures) == 1:
        shape, dtype = next(iter(signatures))
        detail["joint_shape"], detail["joint_dtype"] = list(shape), dtype
    elif signatures:
        add_error(section, f"{label}: joint_action shape/dtype mismatch")
    return detail


def _audit_file(
    path: Path,
    task: str,
    records: Mapping[int, Mapping[str, Any]],
    episodes: Sequence[int],
    exact_episodes: bool,
    label: str,
) -> dict[str, Any]:
    section: dict[str, Any] = {
        "label": label,
        "task": task,
        "path": str(path),
        "episodes": [],
        "completed_count": 0,
        "joint_vector_count": 0,
        "joint_element_count": 0,
        "error_count": 0,
        "errors": [],
    }
    if not path.is_file():
        add_error(section, f"{label}/{task}: HDF5 does not exist")
        return section
    try:
        with h5py.File(path, "r") as handle:
            groups = episode_groups(handle, f"{label}/{task}", section)
            expected = set(episodes)
            actual = set(groups)
            if not expected.issubset(actual):
                add_error(section, f"{label}/{task}: missing episodes {sorted(expected - actual)}")
            if exact_episodes and actual != expected:
                add_error(section, f"{label}/{task}: episode set does not exactly match")
            for episode in episodes:
                if episode not in groups:
                    continue
                detail = _audit_episode(
                    groups[episode],
                    task,
                    episode,
                    records[episode],
                    f"{label}/{task}/episode_{episode}",
                    section,
                )
                section["episodes"].append(detail)
                if detail["final_is_completed"] is True:
                    section["completed_count"] += 1
    except OSError as exc:
        add_error(section, f"{label}/{task}: unable to read HDF5: {exc}")
    return section


def _audit_metadata(
    path: Path,
    task: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    section: dict[str, Any] = {
        "task": task,
        "path": str(path),
        "error_count": 0,
        "errors": [],
    }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        add_error(section, f"{task}: metadata could not be read: {exc}")
        return section
    if not isinstance(payload, Mapping) or payload.get("env_id") != task:
        add_error(section, f"{task}: metadata env_id mismatch")
    if payload.get("record_count") != len(records):
        add_error(section, f"{task}: metadata record_count mismatch")
    if payload.get("records") != list(records):
        add_error(section, f"{task}: metadata records do not match train metadata")
    return section


def _summary(audits: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    keys = ("completed_count", "joint_vector_count", "joint_element_count", "error_count")
    return {
        "file_count": len(audits),
        "episode_count": sum(len(audit["episodes"]) for audit in audits),
        **{key: sum(int(audit.get(key, 0)) for audit in audits) for key in keys},
    }


def validate_generated_dataset_contract(
    generated_root: str | Path,
    tasks: Sequence[str],
    episodes: Sequence[int],
    *,
    records_by_task: Mapping[str, Mapping[int, Mapping[str, Any]]] | None = None,
    reference_root: str | Path = REFERENCE_ROOT,
    metadata_root: str | Path = METADATA_ROOT,
) -> dict[str, Any]:
    """Validate the complete HDF5 contract for generated output and official reference data within the requested scope."""
    output = Path(generated_root).expanduser().resolve()
    reference = Path(reference_root).expanduser().resolve()
    ordered_tasks = list(tasks)
    episode_indices = list(episodes)
    if not ordered_tasks or len(ordered_tasks) != len(set(ordered_tasks)):
        raise DatasetContractError("validation tasks must be non-empty and unique")
    if any(task not in ALL_TASKS for task in ordered_tasks):
        raise DatasetContractError("validation tasks contain unknown environments")
    if not episode_indices or episode_indices != list(range(len(episode_indices))):
        raise DatasetContractError("validation episodes must be a contiguous range starting at 0")
    if records_by_task is None:
        records_by_task = read_train_metadata(metadata_root)

    generated, official, metadata = [], [], []
    for task in ordered_tasks:
        if task not in records_by_task:
            raise DatasetContractError(f"missing train metadata for {task}")
        records_for_task = records_by_task[task]
        if any(episode not in records_for_task for episode in episode_indices):
            raise DatasetContractError(f"{task}: train metadata is missing requested episodes")
        records = [records_for_task[episode] for episode in episode_indices]
        metadata.append(
            _audit_metadata(
                output / f"record_dataset_{task}_metadata.json",
                task,
                records,
            )
        )
        generated.append(
            _audit_file(
                output / f"record_dataset_{task}.h5",
                task,
                records_for_task,
                episode_indices,
                True,
                "generated",
            )
        )
        official.append(
            _audit_file(
                reference / f"record_dataset_{task}.h5",
                task,
                records_for_task,
                episode_indices,
                False,
                "official",
            )
        )

    generated_summary = _summary(generated)
    official_summary = _summary(official)
    metadata_errors = sum(int(item["error_count"]) for item in metadata)
    expected = len(ordered_tasks) * len(episode_indices)
    scope = {
        "tasks": ordered_tasks,
        "episode_indices": episode_indices,
        "expected_episode_count": expected,
        "full_16x100": (
            ordered_tasks == list(ALL_TASKS)
            and episode_indices == list(range(MAX_EPISODES))
        ),
    }
    passed = (
        metadata_errors == 0
        and generated_summary["error_count"] == 0
        and official_summary["error_count"] == 0
        and generated_summary["completed_count"] == expected
        and official_summary["completed_count"] == expected
    )
    return {
        "passed": passed,
        "scope": scope,
        "metadata": {"error_count": metadata_errors, "audits": metadata},
        "generated": {**generated_summary, "audits": generated},
        "official": {**official_summary, "audits": official},
        "acceptance": {
            "expected_final_completed": expected,
            "generated_final_completed": generated_summary["completed_count"],
            "official_final_completed": official_summary["completed_count"],
        },
    }


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the No-Patch generated-data contract")
    parser.add_argument("--output-dir", required=True, help="Existing generated-data output directory")
    parser.add_argument("--env", "--environment", default="all", help="all or comma-separated environment names")
    parser.add_argument("--episodes", type=int, default=MAX_EPISODES, help="Number of episodes to validate per environment starting at episode 0")
    parser.add_argument("--metadata-root", default=str(METADATA_ROOT), help="Current train metadata directory")
    parser.add_argument("--reference-root", default=str(REFERENCE_ROOT), help="Official HDF5 directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    try:
        if not 1 <= args.episodes <= MAX_EPISODES:
            raise DatasetContractError(f"--episodes must be in 1..{MAX_EPISODES}")
        result = validate_generated_dataset_contract(
            args.output_dir,
            parse_tasks(args.env),
            list(range(args.episodes)),
            metadata_root=args.metadata_root,
            reference_root=args.reference_root,
        )
    except DatasetContractError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
