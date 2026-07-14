"""HDF5 严格比较器的轻量级合成数据测试。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest


def _load_module():
    script_path = Path(__file__).parents[2] / "scripts" / "compare_generated_dataset.py"
    spec = importlib.util.spec_from_file_location("compare_generated_dataset_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _write_task_file(
    path: Path,
    *,
    task_id: str,
    include_reference_extra: bool,
    changed_value: bool = False,
    compression: str | None = "gzip",
) -> None:
    with h5py.File(path, "w") as data:
        data.attrs["format_version"] = np.int64(1)
        episode = data.create_group("episode_0")
        episode.attrs.create(
            "goal",
            f"goal-{task_id}",
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        timestep = episode.create_group("timestep_0")
        values = np.asarray([1.0, np.nan, 3.0], dtype=np.float64)
        if changed_value:
            values[2] += 1e-8
        timestep.create_dataset(
            "values",
            data=values,
            chunks=(2,),
            compression=compression,
            maxshape=(None,),
        )
        timestep.create_dataset(
            "labels",
            data=np.asarray(["alpha", "测试"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        if include_reference_extra:
            extra = data.create_group("episode_9")
            extra.create_dataset("ignored", data=np.arange(4, dtype=np.int16))


def _make_dataset_pair(
    root: Path,
    *,
    changed_task: str | None = None,
    changed_compression: bool = False,
) -> tuple[Path, Path, Path]:
    reference = root / "reference"
    generated = root / "generated"
    report = root / "reports"
    reference.mkdir(parents=True)
    generated.mkdir()
    for task_id in MODULE.EXPECTED_TASK_IDS:
        _write_task_file(
            reference / f"record_dataset_{task_id}.h5",
            task_id=task_id,
            include_reference_extra=True,
        )
        _write_task_file(
            generated / f"record_dataset_{task_id}.h5",
            task_id=task_id,
            include_reference_extra=False,
            changed_value=task_id == changed_task,
            compression=None if changed_compression and task_id == changed_task else "gzip",
        )
    return reference, generated, report


@pytest.mark.lightweight
def test_selected_episode_strict_match_ignores_reference_extra_episode(tmp_path: Path) -> None:
    reference, generated, report = _make_dataset_pair(tmp_path)

    summary = MODULE.compare_generated_dataset(
        reference,
        generated,
        report,
        episodes=(0,),
        workspace_root=tmp_path,
    )

    assert summary["passed"] is True
    assert summary["counts"]["tasks_strict_equal"] == 16
    assert summary["counts"]["selected_episodes_strict_equal"] == 16
    assert summary["counts"]["differences"] == 0
    assert all(task["reference_extra_episodes_ignored"] == [9] for task in summary["tasks"])
    assert (report / "differences.jsonl").read_text(encoding="utf-8") == ""
    on_disk = json.loads((report / "comparison.json").read_text(encoding="utf-8"))
    assert on_disk["strict_equal"] is True
    leaf_records = [
        json.loads(line)
        for line in (report / "leaf_comparisons.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    numeric = next(record for record in leaf_records if record["path"].endswith("/values"))
    assert numeric["metrics"]["nan_positions_equal"] is True
    assert numeric["metrics"]["diff_count"] == 0
    assert numeric["reference"]["canonical_sha256"] == numeric["generated"]["canonical_sha256"]


@pytest.mark.lightweight
def test_generated_extra_episode_is_rejected(tmp_path: Path) -> None:
    reference, generated, report = _make_dataset_pair(tmp_path)
    with h5py.File(generated / "record_dataset_BinFill.h5", "a") as data:
        extra = data.create_group("episode_9")
        extra.create_dataset("stale", data=np.int64(1))

    summary = MODULE.compare_generated_dataset(
        reference,
        generated,
        report,
        episodes=(0,),
        workspace_root=tmp_path,
        task_ids=("BinFill",),
        allow_joint_action_allclose=True,
    )

    assert summary["passed"] is False
    assert summary["accepted"] is False
    assert summary["tasks"][0]["generated_extra_episodes"] == [9]
    differences = [
        json.loads(line)
        for line in (report / "differences.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        item["category"] == "unexpected_generated_episode"
        and item["path"] == "/episode_9"
        for item in differences
    )


@pytest.mark.lightweight
def test_numeric_and_storage_differences_are_reported(tmp_path: Path) -> None:
    reference, generated, report = _make_dataset_pair(
        tmp_path,
        changed_task="BinFill",
        changed_compression=True,
    )

    summary = MODULE.compare_generated_dataset(
        reference,
        generated,
        report,
        episodes=(0,),
        workspace_root=tmp_path,
        rtol=1e-7,
    )

    assert summary["passed"] is False
    differences = [
        json.loads(line)
        for line in (report / "differences.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    content = next(item for item in differences if item["category"] == "dataset_content")
    assert content["path"] == "/episode_0/timestep_0/values"
    assert content["metrics"]["strict_equal"] is False
    assert content["metrics"]["allclose"] is True
    assert content["metrics"]["diff_count"] == 1
    metadata = [item for item in differences if item["category"] == "dataset_metadata"]
    assert any(item["field"] == "compression" for item in metadata)


@pytest.mark.lightweight
def test_joint_action_allclose_policy_keeps_strict_result_separate(tmp_path: Path) -> None:
    reference, generated, report = _make_dataset_pair(tmp_path)
    for directory, delta in ((reference, 0.0), (generated, 1e-12)):
        with h5py.File(directory / "record_dataset_BinFill.h5", "a") as data:
            action = data["episode_0/timestep_0"].create_group("action")
            action.create_dataset(
                "joint_action",
                data=np.asarray([1e-6 + delta], dtype=np.float64),
            )

    summary = MODULE.compare_generated_dataset(
        reference,
        generated,
        report,
        episodes=(0,),
        workspace_root=tmp_path,
        task_ids=("BinFill",),
        rtol=1e-5,
        allow_joint_action_allclose=True,
        joint_action_max_abs_diff=2e-12,
    )

    assert summary["strict_equal"] is False
    assert summary["passed"] is False
    assert summary["accepted"] is True
    assert summary["acceptance_policy"]["allowed_difference_count"] == 1
    assert summary["acceptance_policy"]["rejected_difference_count"] == 0


@pytest.mark.lightweight
def test_joint_action_allclose_policy_enforces_absolute_cap(tmp_path: Path) -> None:
    reference, generated, report = _make_dataset_pair(tmp_path)
    for directory, delta in ((reference, 0.0), (generated, 1e-8)):
        with h5py.File(directory / "record_dataset_BinFill.h5", "a") as data:
            action = data["episode_0/timestep_0"].create_group("action")
            action.create_dataset(
                "joint_action",
                data=np.asarray([1.0 + delta], dtype=np.float64),
            )

    summary = MODULE.compare_generated_dataset(
        reference,
        generated,
        report,
        episodes=(0,),
        workspace_root=tmp_path,
        task_ids=("BinFill",),
        rtol=1e-7,
        allow_joint_action_allclose=True,
        joint_action_max_abs_diff=1e-12,
    )

    assert summary["passed"] is False
    assert summary["accepted"] is False
    assert summary["acceptance_policy"]["allowed_difference_count"] == 0
    assert summary["acceptance_policy"]["rejected_difference_count"] == 1


@pytest.mark.lightweight
def test_rejects_outside_workspace_and_symbolic_links(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reference, generated, _ = _make_dataset_pair(workspace)
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(MODULE.UnsafePathError, match="仓库外"):
        MODULE.compare_generated_dataset(
            outside,
            generated,
            workspace / "report-outside",
            episodes=(0,),
            workspace_root=workspace,
        )

    linked_generated = workspace / "linked-generated"
    linked_generated.symlink_to(generated, target_is_directory=True)
    with pytest.raises(MODULE.UnsafePathError, match="符号链接"):
        MODULE.compare_generated_dataset(
            reference,
            linked_generated,
            workspace / "report-link",
            episodes=(0,),
            workspace_root=workspace,
        )


@pytest.mark.lightweight
def test_partial_task_scope_ignores_other_known_reference_tasks(tmp_path: Path) -> None:
    reference, generated, report = _make_dataset_pair(tmp_path)
    for task_id in MODULE.EXPECTED_TASK_IDS:
        if task_id != "BinFill":
            (generated / f"record_dataset_{task_id}.h5").unlink()

    summary = MODULE.compare_generated_dataset(
        reference,
        generated,
        report,
        episodes=(0,),
        workspace_root=tmp_path,
        task_ids=("BinFill",),
    )

    assert summary["passed"] is True
    assert len(summary["file_inventory"]["reference_out_of_scope"]) == 15
    assert summary["counts"]["differences"] == 0


@pytest.mark.lightweight
def test_integers_never_use_numeric_tolerance() -> None:
    metrics = MODULE._compare_loaded_values(
        np.asarray([2**60], dtype=np.int64),
        np.asarray([2**60 + 1], dtype=np.int64),
        rtol=1.0,
        atol=1.0,
    )

    assert metrics["strict_equal"] is False
    assert metrics["allclose"] is False
    assert metrics["diff_count"] == 1
    assert metrics["max_abs_diff"] is None


@pytest.mark.lightweight
def test_missing_internal_object_is_reported_without_key_error(tmp_path: Path) -> None:
    reference, generated, report = _make_dataset_pair(tmp_path)
    with h5py.File(generated / "record_dataset_BinFill.h5", "a") as data:
        del data["episode_0/timestep_0/labels"]

    summary = MODULE.compare_generated_dataset(
        reference,
        generated,
        report,
        episodes=(0,),
        workspace_root=tmp_path,
        task_ids=("BinFill",),
    )

    assert summary["passed"] is False
    differences = [
        json.loads(line)
        for line in (report / "differences.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    missing = next(item for item in differences if item["category"] == "missing_object")
    assert missing["path"] == "/episode_0/timestep_0/labels"
    assert missing["generated"] is None


@pytest.mark.lightweight
def test_null_dataspace_and_wrong_episode_kind_are_auditable(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    generated = tmp_path / "generated"
    reference.mkdir()
    generated.mkdir()
    file_name = "record_dataset_BinFill.h5"
    with h5py.File(reference / file_name, "w") as data:
        episode = data.create_group("episode_0")
        episode.create_dataset("null", data=h5py.Empty(np.dtype("float64")))
    with h5py.File(generated / file_name, "w") as data:
        episode = data.create_group("episode_0")
        episode.create_dataset("null", data=h5py.Empty(np.dtype("float64")))

    matching = MODULE.compare_generated_dataset(
        reference,
        generated,
        tmp_path / "report-null",
        episodes=(0,),
        workspace_root=tmp_path,
        task_ids=("BinFill",),
    )
    assert matching["passed"] is True

    with h5py.File(generated / file_name, "w") as data:
        data.create_dataset("episode_0", data=np.int64(0))
    mismatching = MODULE.compare_generated_dataset(
        reference,
        generated,
        tmp_path / "report-kind",
        episodes=(0,),
        workspace_root=tmp_path,
        task_ids=("BinFill",),
    )
    assert mismatching["passed"] is False
    differences = [
        json.loads(line)
        for line in (tmp_path / "report-kind/differences.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    episode_kind = next(item for item in differences if item["category"] == "episode_kind")
    assert episode_kind["reference_kind"] == "group"
    assert episode_kind["generated_kind"] == "dataset"
