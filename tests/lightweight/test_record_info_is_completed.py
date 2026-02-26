# -*- coding: utf-8 -*-
"""
轻量测试：RecordWrapper 的 info/is_completed 判定与写入接线。

运行方式（使用 uv）：
    uv run python tests/lightweight/test_record_info_is_completed.py
"""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests._shared.repo_paths import find_repo_root


def _record_wrapper_path() -> Path:
    repo_root = find_repo_root(__file__)
    return repo_root / "src/robomme/env_record_wrapper/RecordWrapper.py"


def _load_completion_fn_from_ast():
    src_path = _record_wrapper_path()
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(src_path))

    func_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_is_online_subgoal_completed":
            func_node = node
            break

    assert func_node is not None, "未找到 _is_online_subgoal_completed"

    module = ast.Module(body=[func_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, filename=str(src_path), mode="exec"), namespace)
    fn = namespace["_is_online_subgoal_completed"]
    assert callable(fn)
    return fn, source, tree


def _assert_completion_logic(fn) -> None:
    tasks = [{"name": "a"}, {"name": "b"}, {"name": "c"}]

    assert fn(0, tasks) is False
    assert fn(2, tasks) is False
    assert fn(3, tasks) is True
    assert fn(4, tasks) is True

    assert fn(0, []) is False
    assert fn(0, None) is False

    assert fn(None, tasks) is False
    assert fn("abc", tasks) is False
    assert fn("3", tasks) is True


def _assert_source_wired(source: str, tree: ast.AST) -> None:
    has_helper_call = False
    has_info_key = False
    has_h5_dataset = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "_is_online_subgoal_completed":
                has_helper_call = True
            if isinstance(node.func, ast.Attribute) and node.func.attr == "create_dataset":
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "is_completed":
                    has_h5_dataset = True
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value == "is_completed":
                    has_info_key = True

    assert has_helper_call, "RecordWrapper 未调用 _is_online_subgoal_completed"
    assert has_info_key, "record_data['info'] 未包含 is_completed"
    assert has_h5_dataset, "HDF5 写入未包含 info/is_completed"

    assert "is_completed" in source


def _assert_h5_bool_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="record_is_completed_") as tmp:
        h5_path = Path(tmp) / "contract.h5"
        with h5py.File(h5_path, "w") as h5:
            ep = h5.create_group("episode_0")
            ts = ep.create_group("timestep_0")
            info = ts.create_group("info")
            info.create_dataset("is_completed", data=bool(True))

        with h5py.File(h5_path, "r") as h5:
            value = h5["episode_0"]["timestep_0"]["info"]["is_completed"][()]
            assert bool(value) is True


def main() -> None:
    print("\n[TEST] RecordWrapper info/is_completed")
    fn, source, tree = _load_completion_fn_from_ast()

    _assert_completion_logic(fn)
    print("  logic ✓ 在线递进完成判定")

    _assert_source_wired(source, tree)
    print("  wiring ✓ RecordWrapper 已接入 buffer + HDF5 写入")

    _assert_h5_bool_contract()
    print("  hdf5 ✓ bool 字段契约可读")

    print("\nPASS: record info is_completed tests passed")


if __name__ == "__main__":
    main()
