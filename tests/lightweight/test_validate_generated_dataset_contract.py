"""生成后数据契约验证器的合成数据测试。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import pytest


def _load_module():
    script_path = (
        Path(__file__).parents[2]
        / "scripts"
        / "validate_generated_dataset_contract.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_generated_dataset_contract_test", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _record(episode: int, seed: int, difficulty: str) -> dict[str, object]:
    return {
        "task": "TinyEnv",
        "episode": episode,
        "seed": seed,
        "difficulty": difficulty,
    }


def _write_metadata(
    path: Path,
    records: list[dict[str, object]],
    *,
    record_count: int | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "env_id": "TinyEnv",
                "record_count": len(records) if record_count is None else record_count,
                "records": records,
            }
        ),
        encoding="utf-8",
    )


def _write_h5(
    path: Path,
    records: list[dict[str, object]],
    *,
    difficulty_overrides: dict[int, str] | None = None,
    timestep_indices: dict[int, list[int]] | None = None,
    final_completed: dict[int, bool] | None = None,
    missing_final_completed: set[int] | None = None,
    final_simple_subgoal: dict[int, str] | None = None,
) -> None:
    difficulty_overrides = difficulty_overrides or {}
    timestep_indices = timestep_indices or {}
    final_completed = final_completed or {}
    missing_final_completed = missing_final_completed or set()
    final_simple_subgoal = final_simple_subgoal or {}
    with h5py.File(path, "w") as data:
        for record in records:
            episode_index = int(record["episode"])
            episode = data.create_group(f"episode_{episode_index}")
            setup = episode.create_group("setup")
            setup.create_dataset("seed", data=record["seed"])
            setup.create_dataset(
                "difficulty",
                data=difficulty_overrides.get(
                    episode_index, str(record["difficulty"])
                ),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            indices = timestep_indices.get(episode_index, [0, 1])
            final_index = max(indices) if indices else None
            for timestep_index in indices:
                timestep = episode.create_group(f"timestep_{timestep_index}")
                info = timestep.create_group("info")
                is_final = timestep_index == final_index
                if not is_final or episode_index not in missing_final_completed:
                    info.create_dataset(
                        "is_completed",
                        data=(
                            final_completed.get(episode_index, True)
                            if is_final
                            else False
                        ),
                    )
                info.create_dataset(
                    "simple_subgoal",
                    data=(
                        final_simple_subgoal.get(
                            episode_index, "All tasks completed"
                        )
                        if is_final
                        else "working"
                    ),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )


def _make_workspace(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, list[dict[str, object]]]:
    workspace = tmp_path / "workspace"
    generated = workspace / "generated"
    metadata = workspace / "metadata"
    report = workspace / "reports" / "contract.json"
    generated.mkdir(parents=True)
    metadata.mkdir()
    records = [_record(0, 100, "easy"), _record(1, 200, "medium")]
    return workspace, generated, metadata, report, records


def _write_valid_fixture(
    generated: Path,
    metadata: Path,
    records: list[dict[str, object]],
) -> None:
    _write_h5(generated / "record_dataset_TinyEnv.h5", records)
    _write_metadata(
        generated / "record_dataset_TinyEnv_metadata.json", records
    )
    _write_metadata(
        metadata / "record_dataset_TinyEnv_metadata.json",
        records + [_record(2, 300, "hard")],
    )


@pytest.mark.lightweight
def test_valid_contract_accepts_source_metadata_extra_episodes(tmp_path: Path) -> None:
    workspace, generated, metadata, report, records = _make_workspace(tmp_path)
    _write_valid_fixture(generated, metadata, records)

    result = MODULE.validate_generated_dataset_contract(
        generated, metadata, workspace, report
    )

    assert result["passed"] is True
    assert result["counts"] == {
        "h5_files": 1,
        "environments": 1,
        "episodes": 2,
        "timesteps": 4,
        "errors": 0,
    }
    assert result["environments"][0]["episode_indices"] == [0, 1]
    assert result["environments"][0]["timestep_count"] == 4
    assert [
        episode["timestep_count"]
        for episode in result["environments"][0]["episodes"]
    ] == [2, 2]
    assert json.loads(report.read_text(encoding="utf-8"))["passed"] is True


@pytest.mark.lightweight
def test_reports_metadata_and_h5_setup_value_mismatches(tmp_path: Path) -> None:
    workspace, generated, metadata, report, records = _make_workspace(tmp_path)
    generated_records = [records[0], _record(1, 999, "medium")]
    _write_h5(
        generated / "record_dataset_TinyEnv.h5",
        records,
        difficulty_overrides={1: "hard"},
    )
    _write_metadata(
        generated / "record_dataset_TinyEnv_metadata.json",
        generated_records,
        record_count=3,
    )
    _write_metadata(metadata / "record_dataset_TinyEnv_metadata.json", records)

    result = MODULE.validate_generated_dataset_contract(
        generated, metadata, workspace, report
    )
    codes = [error["code"] for error in result["errors"]]

    assert result["passed"] is False
    assert "metadata_record_count_mismatch" in codes
    assert "generated_metadata_value_mismatch" in codes
    assert "h5_setup_value_mismatch" in codes


@pytest.mark.lightweight
def test_rejects_non_contiguous_h5_and_metadata_episode_set(tmp_path: Path) -> None:
    workspace, generated, metadata, report, records = _make_workspace(tmp_path)
    h5_records = [records[0], _record(2, 300, "hard")]
    _write_h5(generated / "record_dataset_TinyEnv.h5", h5_records)
    _write_metadata(
        generated / "record_dataset_TinyEnv_metadata.json", [records[0]]
    )
    _write_metadata(
        metadata / "record_dataset_TinyEnv_metadata.json", h5_records
    )

    result = MODULE.validate_generated_dataset_contract(
        generated, metadata, workspace, report
    )
    codes = {error["code"] for error in result["errors"]}

    assert "h5_episode_sequence_not_contiguous" in codes
    assert "generated_metadata_episode_set_mismatch" in codes


@pytest.mark.lightweight
def test_rejects_episode_without_any_timestep(tmp_path: Path) -> None:
    workspace, generated, metadata, report, records = _make_workspace(tmp_path)
    _write_h5(
        generated / "record_dataset_TinyEnv.h5",
        records,
        timestep_indices={0: []},
    )
    _write_metadata(
        generated / "record_dataset_TinyEnv_metadata.json", records
    )
    _write_metadata(metadata / "record_dataset_TinyEnv_metadata.json", records)

    result = MODULE.validate_generated_dataset_contract(
        generated, metadata, workspace, report
    )

    assert any(error["code"] == "h5_no_timesteps" for error in result["errors"])
    assert result["counts"]["timesteps"] == 2
    assert result["environments"][0]["episodes"][0]["timestep_count"] == 0


@pytest.mark.lightweight
def test_rejects_non_contiguous_timestep_sequence(tmp_path: Path) -> None:
    workspace, generated, metadata, report, records = _make_workspace(tmp_path)
    _write_h5(
        generated / "record_dataset_TinyEnv.h5",
        records,
        timestep_indices={0: [0, 2]},
    )
    _write_metadata(
        generated / "record_dataset_TinyEnv_metadata.json", records
    )
    _write_metadata(metadata / "record_dataset_TinyEnv_metadata.json", records)

    result = MODULE.validate_generated_dataset_contract(
        generated, metadata, workspace, report
    )
    difference = next(
        error
        for error in result["errors"]
        if error["code"] == "h5_timestep_sequence_not_contiguous"
    )

    assert difference["expected"] == [0, 1, 2]
    assert difference["actual"] == [0, 2]


@pytest.mark.lightweight
def test_rejects_timestep_that_is_not_a_hard_link_group(tmp_path: Path) -> None:
    workspace, generated, metadata, report, records = _make_workspace(tmp_path)
    _write_valid_fixture(generated, metadata, records)
    h5_path = generated / "record_dataset_TinyEnv.h5"
    with h5py.File(h5_path, "a") as data:
        episode = data["episode_0"]
        del episode["timestep_0"]
        target = episode.create_group("hidden_timestep_target")
        target.create_group("info")
        episode["timestep_0"] = h5py.SoftLink(
            "/episode_0/hidden_timestep_target"
        )

    result = MODULE.validate_generated_dataset_contract(
        generated, metadata, workspace, report
    )

    assert any(
        error["code"] == "h5_timestep_not_group" for error in result["errors"]
    )


@pytest.mark.lightweight
def test_rejects_final_false_missing_completion_and_wrong_subgoal(
    tmp_path: Path,
) -> None:
    workspace, generated, metadata, report, records = _make_workspace(tmp_path)
    _write_h5(
        generated / "record_dataset_TinyEnv.h5",
        records,
        final_completed={0: False},
        missing_final_completed={1},
        final_simple_subgoal={0: "still working"},
    )
    _write_metadata(
        generated / "record_dataset_TinyEnv_metadata.json", records
    )
    _write_metadata(metadata / "record_dataset_TinyEnv_metadata.json", records)

    result = MODULE.validate_generated_dataset_contract(
        generated, metadata, workspace, report
    )
    codes = {error["code"] for error in result["errors"]}

    assert "h5_final_not_completed" in codes
    assert "h5_final_is_completed_missing" in codes
    assert "h5_final_simple_subgoal_mismatch" in codes


@pytest.mark.lightweight
def test_rejects_truthy_integer_instead_of_strict_bool_completion(
    tmp_path: Path,
) -> None:
    workspace, generated, metadata, report, records = _make_workspace(tmp_path)
    _write_valid_fixture(generated, metadata, records)
    h5_path = generated / "record_dataset_TinyEnv.h5"
    with h5py.File(h5_path, "a") as data:
        info = data["episode_0/timestep_1/info"]
        del info["is_completed"]
        info.create_dataset("is_completed", data=1)

    result = MODULE.validate_generated_dataset_contract(
        generated, metadata, workspace, report
    )

    assert any(
        error["code"] == "h5_final_is_completed_not_bool"
        for error in result["errors"]
    )


@pytest.mark.lightweight
def test_rejects_missing_generated_metadata_and_temp_residue(tmp_path: Path) -> None:
    workspace, generated, metadata, report, records = _make_workspace(tmp_path)
    _write_h5(generated / "record_dataset_TinyEnv.h5", records)
    _write_metadata(metadata / "record_dataset_TinyEnv_metadata.json", records)
    (generated / "temp_TinyEnv_episode_1.h5").write_bytes(b"incomplete")

    result = MODULE.validate_generated_dataset_contract(
        generated, metadata, workspace, report
    )
    codes = {error["code"] for error in result["errors"]}

    assert "temporary_entry" in codes
    assert "generated_metadata_missing" in codes


@pytest.mark.lightweight
def test_rejects_input_directory_symlink_and_writes_failure_report(
    tmp_path: Path,
) -> None:
    workspace, generated, metadata, report, records = _make_workspace(tmp_path)
    _write_valid_fixture(generated, metadata, records)
    linked_generated = workspace / "linked-generated"
    linked_generated.symlink_to(generated, target_is_directory=True)

    result = MODULE.validate_generated_dataset_contract(
        linked_generated, metadata, workspace, report
    )

    assert result["passed"] is False
    assert result["errors"][0]["code"] == "unsafe_input_path"
    assert "符号链接" in result["errors"][0]["message"]
    assert report.is_file()


@pytest.mark.lightweight
def test_rejects_symlink_inside_generated_tree(tmp_path: Path) -> None:
    workspace, generated, metadata, report, records = _make_workspace(tmp_path)
    _write_valid_fixture(generated, metadata, records)
    (generated / "linked_metadata.json").symlink_to(
        metadata / "record_dataset_TinyEnv_metadata.json"
    )

    result = MODULE.validate_generated_dataset_contract(
        generated, metadata, workspace, report
    )

    assert result["passed"] is False
    assert any(error["code"] == "symbolic_link" for error in result["errors"])


@pytest.mark.lightweight
def test_rejects_outside_input_and_outside_report(tmp_path: Path) -> None:
    workspace, generated, metadata, report, records = _make_workspace(tmp_path)
    _write_valid_fixture(generated, metadata, records)
    outside = tmp_path / "outside"
    outside.mkdir()

    result = MODULE.validate_generated_dataset_contract(
        outside, metadata, workspace, report
    )
    assert result["passed"] is False
    assert result["errors"][0]["code"] == "unsafe_input_path"
    assert "仓库外" in result["errors"][0]["message"]

    with pytest.raises(MODULE.UnsafePathError, match="仓库内"):
        MODULE.validate_generated_dataset_contract(
            generated,
            metadata,
            workspace,
            tmp_path / "outside-report.json",
        )
    with pytest.raises(MODULE.UnsafePathError, match="仓库内"):
        MODULE.validate_generated_dataset_contract(
            generated, metadata, workspace, workspace
        )

    dangling_report = workspace / "reports" / "dangling.json"
    dangling_report.symlink_to(workspace / "不存在.json")
    with pytest.raises(MODULE.UnsafePathError, match="普通文件"):
        MODULE.validate_generated_dataset_contract(
            generated, metadata, workspace, dangling_report
        )


@pytest.mark.lightweight
def test_report_never_creates_a_directory_inside_read_only_inputs(
    tmp_path: Path,
) -> None:
    workspace, generated, metadata, _, records = _make_workspace(tmp_path)
    _write_valid_fixture(generated, metadata, records)
    forbidden_parent = generated / "new-report-directory"

    with pytest.raises(MODULE.UnsafePathError, match="只读输入"):
        MODULE.validate_generated_dataset_contract(
            generated,
            metadata,
            workspace,
            forbidden_parent / "contract.json",
        )

    assert not forbidden_parent.exists()


@pytest.mark.lightweight
def test_cli_exit_status_distinguishes_success_and_contract_failure(
    tmp_path: Path,
) -> None:
    workspace, generated, metadata, report, records = _make_workspace(tmp_path)
    _write_valid_fixture(generated, metadata, records)
    arguments = [
        "--generated-dir",
        str(generated),
        "--metadata-root",
        str(metadata),
        "--workspace-root",
        str(workspace),
        "--report",
        str(report),
    ]

    assert MODULE.main(arguments) == 0
    (generated / "temp_unfinished").mkdir()
    assert MODULE.main(arguments) == 1
