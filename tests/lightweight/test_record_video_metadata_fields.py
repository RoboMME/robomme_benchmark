# -*- coding: utf-8 -*-
"""
Lightweight test: RecordWrapper video-related metadata field wiring (buffer + HDF5 writing).

Run (using uv):
    uv run python tests/lightweight/test_record_video_metadata_fields.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests._shared.repo_paths import find_repo_root

pytestmark = [pytest.mark.lightweight, pytest.mark.gpu]


def _record_wrapper_path() -> Path:
    repo_root = find_repo_root(__file__)
    return repo_root / "src/robomme/env_record_wrapper/RecordWrapper.py"


def _load_source_tree() -> tuple[str, ast.AST]:
    src_path = _record_wrapper_path()
    source = src_path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(src_path))


def _collect_dict_keys(tree: ast.AST) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)
    return keys


def _collect_create_dataset_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "create_dataset":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            names.add(first_arg.value)
    return names


def _run_assertions() -> None:
    source, tree = _load_source_tree()
    dict_keys = _collect_dict_keys(tree)
    dataset_names = _collect_create_dataset_names(tree)

    required_record_keys = {
        "choice_action",
        "simple_subgoal",
        "simple_subgoal_online",
        "grounded_subgoal",
        "grounded_subgoal_online",
        "is_completed",
        "is_subgoal_boundary",
    }
    missing_record_keys = sorted(required_record_keys - dict_keys)
    assert not missing_record_keys, f"record_data missing fields: {missing_record_keys}"
    print("  buffer ✓ record_data contains target fields")

    required_h5_datasets = {
        "choice_action",
        "simple_subgoal",
        "simple_subgoal_online",
        "grounded_subgoal",
        "grounded_subgoal_online",
        "is_completed",
        "is_subgoal_boundary",
    }
    missing_h5_datasets = sorted(required_h5_datasets - dataset_names)
    assert not missing_h5_datasets, f"HDF5 writing missing fields: {missing_h5_datasets}"
    print("  hdf5 ✓ create_dataset already contains target fields")

    # Video superimposed text should directly display the schema field name, facilitating manual verification of recording results.
    for token in [
        "info.simple_subgoal:",
        "info.simple_subgoal_online:",
        "info.grounded_subgoal:",
        "info.grounded_subgoal_online:",
        "action.choice_action:",
        "info.is_completed:",
    ]:
        assert token in source, f"Video superimposed text missing field label: {token}"


def test_record_video_metadata_fields_pytest() -> None:
    _run_assertions()


def main() -> None:
    print("\n[TEST] RecordWrapper video metadata fields")
    _run_assertions()
    print("  video ✓ Superimposed text contains field name labels")

    print("\nPASS: record video metadata fields tests passed")


if __name__ == "__main__":
    main()
