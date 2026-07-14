"""Audit the downloaded RoboMME reference HDF5 dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py

EXPECTED_TASK_IDS = (
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)
EXPECTED_EPISODES_PER_TASK = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_keys(data: h5py.File) -> list[str]:
    return sorted(
        (key for key in data.keys() if key.startswith("episode_")),
        key=lambda key: int(key.removeprefix("episode_")),
    )


def audit_reference_dataset(
    h5_data_dir: str,
    source_revision: str,
    report: str,
) -> None:
    """Validate the official reference dataset and write a JSON evidence report."""
    dataset_dir = Path(h5_data_dir).resolve()
    report_path = Path(report).resolve()
    errors: list[str] = []
    file_records: list[dict[str, object]] = []

    if dataset_dir.is_symlink():
        errors.append(f"dataset directory is a symbolic link: {dataset_dir}")
    if not dataset_dir.is_dir():
        errors.append(f"dataset directory does not exist: {dataset_dir}")
        h5_paths: list[Path] = []
        symlinks: list[str] = []
    else:
        h5_paths = sorted(dataset_dir.glob("*.h5"))
        symlinks = [str(path) for path in dataset_dir.rglob("*") if path.is_symlink()]
        if symlinks:
            errors.append(f"dataset contains symbolic links: {symlinks}")

    expected_names = {f"record_dataset_{task_id}.h5" for task_id in EXPECTED_TASK_IDS}
    actual_names = {path.name for path in h5_paths}
    missing_names = sorted(expected_names - actual_names)
    unexpected_names = sorted(actual_names - expected_names)
    if missing_names:
        errors.append(f"missing expected HDF5 files: {missing_names}")
    if unexpected_names:
        errors.append(f"unexpected HDF5 files: {unexpected_names}")
    if len(h5_paths) != len(EXPECTED_TASK_IDS):
        errors.append(
            f"expected {len(EXPECTED_TASK_IDS)} HDF5 files, found {len(h5_paths)}"
        )

    total_episodes = 0
    for task_id in EXPECTED_TASK_IDS:
        file_path = dataset_dir / f"record_dataset_{task_id}.h5"
        record: dict[str, object] = {
            "task_id": task_id,
            "path": str(file_path),
            "exists": file_path.is_file(),
        }
        if not file_path.is_file():
            file_records.append(record)
            continue

        record["bytes"] = file_path.stat().st_size
        record["sha256"] = _sha256(file_path)
        try:
            with h5py.File(file_path, "r") as data:
                episodes = _episode_keys(data)
                record["episode_count"] = len(episodes)
                record["episode_keys"] = episodes
        except OSError as exc:
            errors.append(f"{task_id}: unreadable HDF5 ({type(exc).__name__}: {exc})")
            record["read_error"] = f"{type(exc).__name__}: {exc}"
        else:
            total_episodes += int(record["episode_count"])
            if record["episode_count"] != EXPECTED_EPISODES_PER_TASK:
                errors.append(
                    f"{task_id}: expected {EXPECTED_EPISODES_PER_TASK} episodes, "
                    f"found {record['episode_count']}"
                )
        file_records.append(record)

    expected_total_episodes = len(EXPECTED_TASK_IDS) * EXPECTED_EPISODES_PER_TASK
    if total_episodes != expected_total_episodes:
        errors.append(
            f"expected {expected_total_episodes} total episodes, found {total_episodes}"
        )

    report_payload = {
        "source_repository": "Yinpei/robomme_data_h5",
        "source_revision": source_revision,
        "dataset_directory": str(dataset_dir),
        "expected_task_count": len(EXPECTED_TASK_IDS),
        "actual_h5_file_count": len(h5_paths),
        "expected_episodes_per_task": EXPECTED_EPISODES_PER_TASK,
        "expected_total_episodes": expected_total_episodes,
        "actual_total_episodes": total_episodes,
        "contains_symbolic_links": bool(symlinks),
        "symbolic_links": symlinks,
        "total_h5_bytes": sum(path.stat().st_size for path in h5_paths),
        "files": file_records,
        "errors": errors,
        "passed": not errors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if errors:
        raise SystemExit(
            f"Reference dataset audit failed; report written to {report_path}"
        )
    print(f"Reference dataset audit passed; report written to {report_path}")


if __name__ == "__main__":
    import tyro

    tyro.cli(audit_reference_dataset)
