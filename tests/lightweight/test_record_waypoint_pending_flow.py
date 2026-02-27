# -*- coding: utf-8 -*-
"""
轻量测试：RecordWrapper waypoint pending 刷新流程。

运行方式（使用 uv）：
    uv run python tests/lightweight/test_record_waypoint_pending_flow.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests._shared.repo_paths import find_repo_root


def _record_wrapper_path() -> Path:
    repo_root = find_repo_root(__file__)
    return repo_root / "src/robomme/env_record_wrapper/RecordWrapper.py"


def _load_tree() -> tuple[ast.Module, str]:
    src_path = _record_wrapper_path()
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(src_path))
    return tree, source


def _find_wrapper_class(tree: ast.Module) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RobommeRecordWrapper":
            return node
    raise AssertionError("未找到 RobommeRecordWrapper")


def _find_method(cls_node: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in cls_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"未找到方法 {method_name}")


def _is_refresh_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
        and func.attr == "_refresh_pending_waypoint"
    )


def _is_super_step_call(node: ast.Call) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "step":
        return False
    value = func.value
    return isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "super"


def _assert_pending_not_cleared(tree: ast.Module, source: str) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or node.value.value is not None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == "_pending_waypoint":
                raise AssertionError("检测到 _pending_waypoint = None，期望不再清空 pending waypoint")
    assert "_pending_waypoint = None" not in source


def _assert_step_refresh_before_super_step(step_fn: ast.FunctionDef) -> None:
    refresh_lines: list[int] = []
    super_step_lines: list[int] = []
    for node in ast.walk(step_fn):
        if not isinstance(node, ast.Call):
            continue
        if _is_refresh_call(node):
            refresh_lines.append(node.lineno)
        if _is_super_step_call(node):
            super_step_lines.append(node.lineno)

    assert refresh_lines, "step() 未调用 self._refresh_pending_waypoint"
    assert super_step_lines, "step() 未调用 super().step"
    assert min(refresh_lines) < min(super_step_lines), (
        "step() 中 _refresh_pending_waypoint 应位于 super().step 之前"
    )


def _assert_close_no_trailing_consume(close_fn: ast.FunctionDef) -> None:
    banned_attrs = {
        "_refresh_pending_waypoint",
        "_consume_pending_waypoint",
        "backfill_waypoint_actions_in_buffer",
    }
    for node in ast.walk(close_fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in banned_attrs:
            raise AssertionError(
                f"close() 不应调用 {func.attr}，当前检测到 trailing waypoint 后处理逻辑"
            )
        if isinstance(func, ast.Name) and func.id in banned_attrs:
            raise AssertionError(
                f"close() 不应调用 {func.id}，当前检测到 trailing waypoint 后处理逻辑"
            )


def main() -> None:
    print("\n[TEST] RecordWrapper waypoint pending flow")
    tree, source = _load_tree()
    cls_node = _find_wrapper_class(tree)

    _assert_pending_not_cleared(tree, source)
    print("  pending ✓ 不再清空 _pending_waypoint")

    step_fn = _find_method(cls_node, "step")
    _assert_step_refresh_before_super_step(step_fn)
    print("  step ✓ _refresh_pending_waypoint 在 super().step 之前")

    close_fn = _find_method(cls_node, "close")
    _assert_close_no_trailing_consume(close_fn)
    print("  close ✓ 无 trailing consume/backfill 后处理")

    print("\nPASS: record waypoint pending flow tests passed")


if __name__ == "__main__":
    main()
