#!/usr/bin/env python3
"""组装并写出 No-Patch 生成、验证与 joint_action 比较报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from compare_joint_actions import JointActionComparisonError, compare_joint_actions
from validate_generated_dataset_contract import (
    MAX_EPISODES,
    METADATA_ROOT,
    REFERENCE_ROOT,
    DatasetContractError,
    parse_tasks,
    read_train_metadata,
    validate_generated_dataset_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_ABS_DIFF = 1e-8
SCRIPT_DIR = Path(__file__).resolve().parent
REPORTS_ROOT = SCRIPT_DIR / "reports"


class ReportGenerationError(RuntimeError):
    """报告输入或写入过程不满足 No-Patch 报告契约。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def write_text_atomic(path: Path, value: str) -> None:
    """使用同目录临时文件原子替换，避免中断时留下半份报告。"""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def new_generation_report(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """创建由生成器或只读复核共用的报告基础 provenance。"""
    lock_path = REPO_ROOT / "uv.lock"
    if not lock_path.is_file():
        raise ReportGenerationError(f"缺少 uv.lock: {lock_path}")
    return {
        "schema_version": 2,
        "status": "running",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_head": _head(),
        "uv_lock_sha256": _sha256(lock_path),
        "parameters": dict(parameters),
    }


def build_validation_report(
    output_dir: str | Path,
    tasks: Sequence[str],
    episodes: Sequence[int],
    *,
    records_by_task: Mapping[str, Mapping[int, Mapping[str, Any]]] | None = None,
    reference_root: str | Path = REFERENCE_ROOT,
    metadata_root: str | Path = METADATA_ROOT,
    max_abs_diff: float = MAX_ABS_DIFF,
) -> dict[str, Any]:
    """以同一 scope 组合独立 contract 与 joint_action 比较结果。"""
    contract = validate_generated_dataset_contract(
        output_dir,
        tasks,
        episodes,
        records_by_task=records_by_task,
        reference_root=reference_root,
        metadata_root=metadata_root,
    )
    comparison = compare_joint_actions(
        output_dir,
        reference_root,
        tasks,
        episodes,
        max_abs_diff=max_abs_diff,
    )
    acceptance = {
        **dict(contract["acceptance"]),
        "max_allowed_abs_diff": float(max_abs_diff),
        "max_abs_diff": comparison["max_abs_diff"],
    }
    passed = bool(contract["passed"] and comparison["passed"])
    return {
        "passed": passed,
        "scope": contract["scope"],
        "metadata": contract["metadata"],
        "generated": contract["generated"],
        "official": contract["official"],
        "joint_action_comparison": comparison,
        "acceptance": acceptance,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _value(value: Any) -> str:
    return "未知" if value is None else str(value)


def _collect_errors(report: Mapping[str, Any], limit: int = 12) -> list[str]:
    """从结构化报告提取少量错误摘要，JSON 仍保留完整错误列表。"""
    messages: list[str] = []
    validation = _mapping(report.get("validation"))
    metadata = _mapping(validation.get("metadata"))
    for audit in metadata.get("audits", []):
        if isinstance(audit, Mapping):
            messages.extend(str(item) for item in audit.get("errors", []))
    for section_name in ("generated", "official"):
        section = _mapping(validation.get(section_name))
        for audit in section.get("audits", []):
            if isinstance(audit, Mapping):
                messages.extend(str(item) for item in audit.get("errors", []))
    comparison = _mapping(validation.get("joint_action_comparison"))
    messages.extend(str(item) for item in comparison.get("errors", []))
    error = _mapping(report.get("error"))
    if error:
        messages.append(f"{error.get('type', 'Error')}: {error.get('message', '')}")
    return messages[:limit]


def render_markdown(report: Mapping[str, Any]) -> str:
    """渲染人类可读的完整摘要，逐轨迹细节保留在 JSON。"""
    validation = _mapping(report.get("validation"))
    scope = _mapping(validation.get("scope"))
    acceptance = _mapping(validation.get("acceptance"))
    comparison = _mapping(validation.get("joint_action_comparison"))
    generation = _mapping(report.get("generation"))
    metadata = _mapping(validation.get("metadata"))
    generated = _mapping(validation.get("generated"))
    official = _mapping(validation.get("official"))
    parameters = _mapping(report.get("parameters"))
    lines = [
        "# No-Patch 数据生成报告",
        "",
        "## 运行来源",
        "",
        f"- 状态：{report.get('status')}",
        f"- 当前 HEAD：{report.get('current_head')}",
        f"- uv.lock SHA-256：{report.get('uv_lock_sha256')}",
        f"- 报告时间（UTC）：{report.get('generated_at_utc')}",
        f"- 报告模式：{report.get('report_mode', 'generation')}",
        "",
        "## 参数",
        "",
    ]
    lines.extend(
        f"    {line}"
        for line in json.dumps(parameters, ensure_ascii=False, indent=2).splitlines()
    )
    lines.extend(
        (
            "",
            "## 范围",
            "",
            f"- 任务数：{len(scope.get('tasks', []))}",
            f"- episode：{scope.get('episode_indices', [])}",
            f"- 期望轨迹数：{scope.get('expected_episode_count', 0)}",
            f"- 完整 16×9：{scope.get('full_16x9', False)}",
            "",
            "## 生成与合约",
            "",
            f"- worker 成功数：{_value(generation.get('success_count'))}",
            f"- worker 失败数：{_value(generation.get('failure_count'))}",
            f"- metadata 错误数：{metadata.get('error_count', 0)}",
            f"- 生成 HDF5 错误数：{generated.get('error_count', 0)}",
            f"- 官方 HDF5 错误数：{official.get('error_count', 0)}",
            f"- 官方最终完成：{acceptance.get('official_final_completed', 0)}/{acceptance.get('expected_final_completed', 0)}",
            f"- 生成最终完成：{acceptance.get('generated_final_completed', 0)}/{acceptance.get('expected_final_completed', 0)}",
            "",
            "## joint_action 逐元素比较",
            "",
            f"- 向量数：{comparison.get('joint_vector_count', 0)}",
            f"- 元素数：{comparison.get('joint_element_count', 0)}",
            f"- 不同元素数：{comparison.get('different_element_count', 0)}",
            f"- 比较错误数：{comparison.get('error_count', 0)}",
            f"- 最大绝对差：{comparison.get('max_abs_diff')}",
            f"- 最大差位置：{comparison.get('max_abs_diff_location')}",
            f"- 最大允许绝对差：{acceptance.get('max_allowed_abs_diff')}",
            f"- 容差内：{comparison.get('within_max_abs_diff', False)}",
            "",
            "## 结论",
            "",
            f"- 完整验收通过：{validation.get('passed', False)}",
        )
    )
    errors = _collect_errors(report)
    if errors:
        lines.extend(("", "## 错误摘要", ""))
        lines.extend(f"- {message}" for message in errors)
        lines.append("- 完整错误与逐轨迹审计请查看同目录 JSON。")
    report_paths = _mapping(report.get("report_paths"))
    if report_paths:
        lines.extend(
            (
                "",
                "## 产物",
                "",
                f"- JSON：{report_paths.get('json')}",
                f"- Markdown：{report_paths.get('markdown')}",
            )
        )
    return "\n".join(lines) + "\n"


def write_generation_report(
    output_dir: str | Path,
    report: dict[str, Any],
) -> dict[str, str]:
    """将统一 JSON/Markdown 写入脚本目录下固定的 reports/。"""
    output = Path(output_dir).expanduser().resolve()
    if not output.is_dir():
        raise ReportGenerationError(f"输出目录不存在: {output}")
    report_dir = REPORTS_ROOT
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "no_patch_generation_report.json"
    markdown_path = report_dir / "no_patch_generation_report.md"
    paths = {"json": str(json_path), "markdown": str(markdown_path)}
    report["report_paths"] = paths
    write_text_atomic(json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_text_atomic(markdown_path, render_markdown(report))
    return paths


def _load_prior_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportGenerationError(f"无法读取 prior report {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ReportGenerationError(f"prior report 必须是 JSON object: {path}")
    return dict(payload)


def build_revalidation_report(
    output_dir: str | Path,
    env: str,
    tasks: Sequence[str],
    episodes: Sequence[int],
    *,
    workers: int | None = None,
    gpus: str | None = None,
    prior_report: Mapping[str, Any] | None = None,
    prior_report_path: str | None = None,
    reference_root: str | Path = REFERENCE_ROOT,
    metadata_root: str | Path = METADATA_ROOT,
    max_abs_diff: float = MAX_ABS_DIFF,
) -> dict[str, Any]:
    """重审已有输出；不会调用 gym、生成 worker 或修改 HDF5/metadata。"""
    output = Path(output_dir).expanduser().resolve()
    prior_parameters = _mapping(prior_report.get("parameters")) if prior_report else {}
    parameters = dict(prior_parameters)
    parameters.update(
        {
            "output_dir": str(output),
            "env": env,
            "tasks": list(tasks),
            "episodes": len(episodes),
            "metadata_root": str(Path(metadata_root).expanduser().resolve()),
            "reference_root": str(Path(reference_root).expanduser().resolve()),
            "max_abs_diff": float(max_abs_diff),
        }
    )
    if workers is not None:
        parameters["workers"] = workers
    if gpus is not None:
        parameters["gpus"] = [item.strip() for item in gpus.split(",") if item.strip()]

    report = new_generation_report(parameters)
    report["report_mode"] = "existing_output_revalidation"
    if prior_report_path is not None:
        report["prior_report"] = prior_report_path
    if prior_report and prior_report.get("current_head"):
        report["prior_generation_head"] = prior_report["current_head"]
    generation = _mapping(prior_report.get("generation")) if prior_report else {}
    report["generation"] = (
        dict(generation)
        if generation
        else {
            "mode": "existing_output_revalidation",
            "requested_count": len(tasks) * len(episodes),
            "success_count": None,
            "failure_count": None,
            "results": [],
        }
    )
    records_by_task = read_train_metadata(metadata_root)
    report["validation"] = build_validation_report(
        output,
        tasks,
        episodes,
        records_by_task=records_by_task,
        reference_root=reference_root,
        metadata_root=metadata_root,
        max_abs_diff=max_abs_diff,
    )
    report["status"] = "passed" if report["validation"]["passed"] else "failed"
    return report


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复核已有 No-Patch 输出，并将最新完整报告固定写入 scripts/data-generation-v2-noPatch/reports/")
    parser.add_argument("--output-dir", required=True, help="已有生成数据输出目录")
    parser.add_argument("--env", "--environment", default="all", help="all 或逗号分隔环境名")
    parser.add_argument("--episodes", type=int, default=MAX_EPISODES, help="每环境复核 0 开始的 episode 数")
    parser.add_argument("--workers", type=int, help="可选：写入报告参数")
    parser.add_argument("--gpus", help="可选：写入报告参数，例如 0,1")
    parser.add_argument("--prior-report", help="可选：保留此前生成报告中的 generation provenance")
    parser.add_argument("--metadata-root", default=str(METADATA_ROOT), help="当前 train metadata 目录")
    parser.add_argument("--reference-root", default=str(REFERENCE_ROOT), help="官方 HDF5 目录")
    parser.add_argument("--max-abs-diff", type=float, default=MAX_ABS_DIFF)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    tasks: list[str] = []
    episodes: list[int] = []
    report: dict[str, Any] | None = None
    try:
        if not 1 <= args.episodes <= MAX_EPISODES:
            raise ReportGenerationError(f"--episodes 必须在 1..{MAX_EPISODES}")
        tasks = parse_tasks(args.env)
        episodes = list(range(args.episodes))
        prior_path = (
            Path(args.prior_report).expanduser().resolve()
            if args.prior_report
            else None
        )
        prior = _load_prior_report(prior_path) if prior_path else None
        report = build_revalidation_report(
            output,
            args.env,
            tasks,
            episodes,
            workers=args.workers,
            gpus=args.gpus,
            prior_report=prior,
            prior_report_path=str(prior_path) if prior_path else None,
            metadata_root=args.metadata_root,
            reference_root=args.reference_root,
            max_abs_diff=args.max_abs_diff,
        )
        paths = write_generation_report(output, report)
    except (
        DatasetContractError,
        JointActionComparisonError,
        ReportGenerationError,
        ValueError,
    ) as exc:
        parameters = {
            "output_dir": str(output),
            "env": args.env,
            "tasks": tasks,
            "episodes": args.episodes,
            "metadata_root": str(args.metadata_root),
            "reference_root": str(args.reference_root),
            "max_abs_diff": args.max_abs_diff,
        }
        try:
            report = new_generation_report(parameters)
            report["report_mode"] = "existing_output_revalidation"
            report["status"] = "failed"
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if output.is_dir():
                paths = write_generation_report(output, report)
            else:
                paths = {}
        except Exception as report_error:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error": f"{exc}; 同时无法写报告: {report_error}",
                    },
                    ensure_ascii=False,
                )
            )
            return 1
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                    "report_paths": paths,
                },
                ensure_ascii=False,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_paths": paths,
                "max_abs_diff": report["validation"]["joint_action_comparison"][
                    "max_abs_diff"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
